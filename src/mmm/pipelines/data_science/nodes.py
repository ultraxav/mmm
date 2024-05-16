"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""

import warnings
from typing import Any, Dict

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM

from ...extras import plots

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
    )

    mmm.sample_posterior_predictive(X, extend_idata=True, combined=True)

    return mmm


def model_diagnostics(mmm: DelayedSaturatedMMM, params: Dict[str, Any]) -> Any:
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
    model_posterior_predictive = plots.posterior_predictive_check_plot(mmm, params)

    # Model Summary Plot
    y_actuals = mmm.idata["fit_data"]["y"].to_numpy()
    y_predicted = (
        mmm.compute_mean_contributions_over_time(original_scale=True)
        .sum(axis=1)
        .to_numpy()
    )
    model_summary_plot = plots.model_summary_plot(y_actuals, y_predicted, "In-Sample")

    return (
        model_summary,
        model_trace,
        model_posterior_predictive,
        model_summary_plot,
    )


def channel_contributions(
    data: pd.DataFrame, mmm: DelayedSaturatedMMM, params: Dict[str, Any]
) -> Any:
    # Model Mean Contributions Over Time
    get_mean_contributions_over_time_df = mmm.compute_mean_contributions_over_time(
        original_scale=True
    ).reset_index()

    # Channel Contribution Breakdown Over Time
    channel_contribution_breakdown = plots.contribution_breakdown_over_time_plot(mmm)

    # Channel Alphas
    channel_alpha = plots.plot_channel_parameter(mmm, "alpha")

    # # Channel Lam
    channel_lam = plots.plot_channel_parameter(mmm, "lam")

    # # Channel Beta
    channel_beta = plots.plot_channel_parameter(mmm, "beta_channel")

    # Channel Contribution Share
    channel_contribution = plots.plot_channel_contribution(
        mmm, params["channel_columns"]
    )

    # Channel Direct Contribution
    channel_direct_contribution = mmm.plot_direct_contribution_curves()
    [ax.set(xlabel="x") for ax in channel_direct_contribution.axes]

    # Channel ROAS
    channel_roas = plots.plot_channel_roas(data, mmm, params["channel_columns"])

    return (
        get_mean_contributions_over_time_df,
        channel_contribution_breakdown,
        channel_alpha,
        channel_lam,
        channel_beta,
        channel_contribution,
        channel_direct_contribution,
        channel_roas,
    )


def out_of_sample_preds(
    data_in_sample: pd.DataFrame,
    data_out_sample: pd.DataFrame,
    mmm: DelayedSaturatedMMM,
    params: Dict[str, Any],
) -> Any:
    data_in_sample = data_in_sample.drop(columns=params["features_to_drop"])
    data_out_sample = data_out_sample.drop(columns=params["features_to_drop"])

    y_out_of_sample = mmm.sample_posterior_predictive(
        X_pred=data_out_sample, extend_idata=False
    )
    y_out_of_sample_with_adstock = mmm.sample_posterior_predictive(
        X_pred=data_out_sample, extend_idata=False, include_last_observations=True
    )

    y_out_of_sample = y_out_of_sample["y"].to_series().groupby("date").mean()
    y_out_of_sample_with_adstock = y_out_of_sample_with_adstock["y"].to_series().groupby("date").mean()

    return fig
