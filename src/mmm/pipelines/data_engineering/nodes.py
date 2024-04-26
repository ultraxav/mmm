"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.14
"""

import pandas as pd


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:

    data = data.rename(columns={"DATE": "date_week"})

    data = pd.concat(
        [data, pd.get_dummies(data["events"], dtype=int).iloc[:, :-1]], axis=1
    )

    data = data.drop(columns=["events"])

    data["day_of_year"] = pd.to_datetime(data["date_week"]).dt.dayofyear

    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:

    data["t"] = range(data.shape[0])

    data["day_of_year"] = data["day_of_year"] / 365

    data_train = data.loc[: int(data.shape[0] * 0.8), :]

    data_test = data.loc[int(data.shape[0] * 0.8) + 1 :, :]

    return data_train, data_test
