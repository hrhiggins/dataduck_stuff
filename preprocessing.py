import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler


# Add RUL values to training and validation datasets
# https://medium.com/@mihaitimoficiuc/predicting-jet-engine-failures-with-nasas-c-mapss-dataset-and-lstm-a-practical-guide-to-85b9513ea9ed
def add_rul_labels(data_set, rul_cap):
    data_set = data_set.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


    rul_values = []
    for engine_id, engine in data_set.groupby("engine_id"):
        highest_cycle = engine["cycle"].max()
        for cycle in engine["cycle"]:
            rul = highest_cycle - cycle
            if rul > rul_cap:
                current_rul = rul_cap
            else:
                current_rul = rul
            rul_values.append(current_rul)

    data_set["rul"] = rul_values
    return data_set


# https://medium.com/@mihaitimoficiuc/predicting-jet-engine-failures-with-nasas-c-mapss-dataset-and-lstm-a-practical-guide-to-85b9513ea9ed
# https://medium.com/@hamalyas_/jet-engine-remaining-useful-life-rul-prediction-a8989d52f194
def remove_non_relevant_data(data_set, variance_threshold, correlation_threshold):
    sensor_cols = get_sensor_columns(data_set)
    # Drop sensors less than variance_threshold
    variances = data_set[sensor_cols].var()
    cols_to_drop = variances[variances < variance_threshold].index.tolist()

    # Calc correlations
    # https://www.geeksforgeeks.org/python/how-to-calculate-correlation-between-two-columns-in-pandas/
    # https: // sparkbyexamples.com / pandas / pandas - correlation - of - columns /
    # https: // stackoverflow.com / questions / 45897003 / python - numpy - corrcoef - runtimewarning - invalid - value - encountered - in -true - divide
    for column in data_set[sensor_cols]:
        # Compute correlation per engine then average
        column_correlation = {"correlation" : 0, "number_columns" : 0}
        for engine_id, engine in data_set.groupby("engine_id"):
            # Avoid spearman warning by checking that the columns are not constant
            if engine[column].nunique() <= 1 or engine['rul'].nunique() <= 1:
                correlation = np.nan
            else:
                # Supress warning of errorstate, NaNs are replaced later
                with np.errstate(divide="ignore", invalid="ignore"):
                    correlation = engine[column].corr(engine["rul"], method="spearman")
            # Robust to NaNs:
            if not pd.isna(correlation):
                column_correlation["correlation"] += abs(correlation)
                column_correlation["number_columns"] += 1

        try:
            average_correlation = column_correlation["correlation"] / column_correlation["number_columns"]
        except ZeroDivisionError:
            average_correlation = 0

        # If NaN produced or value less then threshold
        if pd.isna(average_correlation) or abs(average_correlation) < correlation_threshold:
            if column not in cols_to_drop:
                cols_to_drop.append(column)

    data_set = data_set.drop(columns=cols_to_drop)

    return data_set, cols_to_drop


def get_sensor_columns(data_set):
    sensor_columns = []
    for column in data_set.columns:
        if column.startswith("sensor"):
            sensor_columns.append(column)

    return sensor_columns


def get_data_columns(data_set):
    # RUL needs fault detection labels for inferences so do not exclude them
    non_data_columns = {"engine_id", "cycle", "fault", "rul"}

    data_columns = []

    for column in data_set.columns:
        if column not in non_data_columns:
            data_columns.append(column)

    return data_columns


def add_fault_labels(data_set, sig, healthy_cycles):
    data_set = data_set.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    data_set = smooth_data(data_set, sigma=sig)
    data_cols = get_data_columns(data_set)

    # Add fault column to dataset
    data_set["fault"] = 0

    for engine_id, engine in data_set.groupby("engine_id"):
        values_dict = {}
        for column in data_set[data_cols].columns:
            # https://www.geeksforgeeks.org/pandas/get-first-n-records-of-a-pandas-dataframe/
            values_dict[column] = ((column.iloc[:healthy_cycles, engine.columns.get_loc(column)]).mean())/healthy_cycles


# https://medium.com/@mihaitimoficiuc/predicting-jet-engine-failures-with-nasas-c-mapss-dataset-and-lstm-a-practical-guide-to-85b9513ea9ed
def pad_data_set(data_set, sequence_length):
    data_set_copy = data_set.copy()
    data_set_sorted = data_set_copy.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    number_needed_rows_dict = {}
    new_padded_engines = []

    for engine_id, engine_data in data_set_sorted.groupby("engine_id"):

        if len(engine_data) < sequence_length:
            number_needed_rows = sequence_length - len(engine_data)
            number_needed_rows_dict[engine_id] = number_needed_rows

            padding_df = pd.DataFrame(data=0, index=range(number_needed_rows), columns=engine_data.columns)
            # Add engineID for alignment
            padding_df["engine_id"] = engine_id
            # Negative cycles so no interferance with sorting
            padding_df["cycle"] = -np.arange(number_needed_rows, 0, -1)
            padding_df["rul"] = np.nan
            # Add padding rows to engine
            engine_data = pd.concat([padding_df, engine_data], ignore_index=True)

        new_padded_engines.append(engine_data)

    # Convert to dataframe
    padded_data = pd.concat(new_padded_engines, ignore_index=True)
    padded_data = padded_data.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    return padded_data


#https://www.geeksforgeeks.org/data-science/signal-smoothing-with-scipy/
def smooth_data(data_set, sig):
    data_set = data_set.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    sensor_cols = get_sensor_columns(data_set)

    for engine_id, engine in data_set.groupby("engine_id"):
        for column in sensor_cols:
            smoothed = gaussian_filter1d(engine[column].values, sigma=sig)
            # Set the smoothed data as the data for that column in correct engine
            data_set.loc[engine.index, column] = smoothed

    return data_set


# https://www.geeksforgeeks.org/machine-learning/how-to-normalize-data-using-scikit-learn-in-python/
# https://www.geeksforgeeks.org/python/how-to-scale-pandas-dataframe-columns/
def normalise(primary_data, secondary_data):
    data_columns = get_data_columns(primary_data)
    primary = primary_data.copy()
    secondary = secondary_data.copy()

    # Do not touch padded rows (must stay 0.0 for masking)
    non_padded_primary_data_cols = primary.loc[primary["cycle"] > 0.0, data_columns]
    non_padded_secondary_data_cols = secondary.loc[secondary["cycle"] > 0.0, data_columns]

    scaler = MinMaxScaler()
    scaler.fit(non_padded_primary_data_cols)

    primary_scaled = scaler.transform(non_padded_primary_data_cols)
    secondary_scaled = scaler.transform(non_padded_secondary_data_cols)

    # Specify float as Pandas unhappy otherwise
    primary[data_columns] = primary[data_columns].astype(float)
    secondary[data_columns] = secondary[data_columns].astype(float)

    primary.loc[primary["cycle"] > 0.0, data_columns] = primary_scaled
    secondary.loc[secondary["cycle"] > 0.0, data_columns] = secondary_scaled

    return primary, secondary, scaler


def normalise_test(primary_data, scaler):
    data_columns = get_data_columns(primary_data)
    primary = primary_data.copy()

    non_padded_primary_data_cols = primary.loc[primary["cycle"] > 0.0, data_columns]
    primary_scaled = scaler.transform(non_padded_primary_data_cols)
    primary[data_columns] = primary[data_columns].astype(float)
    primary.loc[primary["cycle"] > 0.0, data_columns] = primary_scaled


    return primary



