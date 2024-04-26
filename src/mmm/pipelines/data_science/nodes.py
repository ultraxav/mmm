"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""

import warnings
from typing import Any, Dict

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM

warnings.filterwarnings("ignore")


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

    mmm.sample_posterior_predictive(X, extend_idata=True, combined=True)

    return mmm


def model_diagnostics(mmm: DelayedSaturatedMMM) -> Any:

    # Model Summary
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

    # Model Trace
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
    model_trace = plt.gcf().suptitle("Model Trace", fontsize=16).get_figure()

    # Model Posterior Predictive Check
    model_posterior_predictive = mmm.plot_posterior_predictive(
        original_scale=True
    ).get_figure()

    # Model Posterior Predictive Components
    model_components_contributions = mmm.plot_components_contributions()

    # Model Contribution Breakdown Over Time
    groups = {
        "Base": [
            "intercept",
            "event_1",
            "event_2",
            # "day_of_year",
            "t",
            "sin_order_1",
            "sin_order_2",
            "cos_order_1",
            "cos_order_2",
        ],
        "tv_S": ["tv_S"],
        "ooh_S": ["ooh_S"],
        "print_S": ["print_S"],
        "search_S": ["search_S"],
        "facebook_S": ["facebook_S"],
        "newsletter": ["newsletter"],
    }
    fig = mmm.plot_grouped_contribution_breakdown_over_time(
        stack_groups=groups,
        original_scale=True,
        # area_kwargs={
        #     "color": {
        #         "tv_S": "C0",
        #         "ooh_S": "C1",
        #         "print_S": "C2",
        #         "search_S": "C3",
        #         "facebook_S": "C4",
        #         "newsletter": "C5",
        #         "Base": "gray",
        #         "Seasonality": "black",
        #     },
        #     "alpha": 0.7,
        # },
    )
    model_contribution_breakdown = fig.suptitle(
        "Contribution Breakdown Over Time", fontsize=16
    ).get_figure()

    # Model Mean Contributions Over Time
    get_mean_contributions_over_time_df = mmm.compute_mean_contributions_over_time(
        original_scale=True
    ).reset_index()

    return (
        model_summary,
        model_trace,
        model_posterior_predictive,
        model_components_contributions,
        model_contribution_breakdown,
        get_mean_contributions_over_time_df,
    )
