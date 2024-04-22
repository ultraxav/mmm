"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""

from typing import Any, Dict

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM


def model_training(data: pd.DataFrame, params: Dict[str, Any]) -> DelayedSaturatedMMM:

    data = data.drop(columns=params["features_to_drop"])

    # # First, let’s compute the share of spend per channel:
    # total_spend_per_channel = data[params['channel_columns']].sum(axis=0)
    # spend_share = total_spend_per_channel / total_spend_per_channel.sum()
    # print(spend_share)
    # print(params)

    # # The scale necessary to make a HalfNormal distribution have unit variance
    # HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)
    # n_channels = len(params['channel_columns'])
    # prior_sigma = HALFNORMAL_SCALE * n_channels * spend_share.to_numpy()
    # print(prior_sigma.tolist())
    # print(params['model_config']['beta_channel'])

    mmm = DelayedSaturatedMMM(
        model_config=params["model_config"],
        sampler_config=params["sampler_config"],
        date_column=params["date_column"],
        channel_columns=params["channel_columns"],
        control_columns=params["control_columns"],
        adstock_max_lag=params["adstock_max_lag"],
        yearly_seasonality=params["yearly_seasonality"],
    )

    X = data.drop(params["objective_variable"], axis=1)
    y = data[params["objective_variable"]]

    mmm.fit(
        X=X,
        y=y,
        target_accept=params["target_accept"],
        chains=params["chains"],
        random_seed=123,
    )

    return mmm


def model_report(mmm: DelayedSaturatedMMM) -> Any:

    model_summary = az.summary(
        data=mmm.fit_result,
        var_names=[
            "intercept",
            "likelihood_sigma",
            "beta_channel",
            "alpha",
            "lam",
            "gamma_control",
            "gamma_fourier",
        ],
    ).reset_index()

    _ = az.plot_trace(
        data=mmm.fit_result,
        var_names=[
            "intercept",
            "likelihood_sigma",
            "beta_channel",
            "alpha",
            "lam",
            "gamma_control",
            "gamma_fourier",
        ],
        compact=True,
        backend_kwargs={"figsize": (12, 10), "layout": "constrained"},
    )

    model_plot = plt.gcf().suptitle("Model Trace", fontsize=16).get_figure()

    return model_summary, model_plot
