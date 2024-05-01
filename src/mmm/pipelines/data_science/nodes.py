"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""

import datetime
import warnings
from typing import Any, Dict

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    model_summary_plot = plots.model_summary_plot(mmm)

    return (
        model_summary,
        model_trace,
        model_posterior_predictive,
        model_summary_plot,
    )


def channel_contributions(mmm: DelayedSaturatedMMM) -> Any:
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
    channel_contribution = mmm.plot_channel_contribution_share_hdi(figsize=(7, 5))

    # Channel Direct Contribution
    channel_direct_contribution = mmm.plot_direct_contribution_curves()
    [ax.set(xlabel="x") for ax in channel_direct_contribution.axes]

    # Channel Contribution function
    channel_contribution_func = mmm.plot_channel_contributions_grid(
        start=0, stop=1.5, num=12
    )
    channel_contribution_func_abs = mmm.plot_channel_contributions_grid(
        start=0, stop=1.5, num=12, absolute_xrange=True
    )

    return (
        get_mean_contributions_over_time_df,
        channel_contribution_breakdown,
        channel_alpha,
        channel_lam,
        channel_beta,
        channel_contribution,
        channel_direct_contribution,
        channel_contribution_func,
        channel_contribution_func_abs,
    )


def channel_roas(
    data: pd.DataFrame, mmm: DelayedSaturatedMMM, params: Dict[str, Any]
) -> Any:
    # ROAS
    channel_contribution_original_scale = (
        mmm.compute_channel_contribution_original_scale()
    )
    roas_samples = (
        channel_contribution_original_scale.stack(sample=("chain", "draw")).sum("date")
        / data[params["channel_columns"]].sum().to_numpy()[..., None]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for channel in params["channel_columns"]:
        sns.histplot(
            roas_samples.sel(channel=channel).to_numpy(),
            binwidth=0.05,
            alpha=0.3,
            kde=True,
            ax=ax,
        )
    ax.set(title="Posterior ROAS distribution", xlabel="ROAS")

    return fig


def out_of_sample_preds(
    data: pd.DataFrame,
    data_test: pd.DataFrame,
    mmm: DelayedSaturatedMMM,
    params: Dict[str, Any],
) -> Any:
    # Data
    data = data.drop(columns=params["features_to_drop"])
    X = data.drop(params["objective_variable"], axis=1)
    y = data[params["objective_variable"]]

    data_test = data_test.drop(columns=params["features_to_drop"])
    X_out_of_sample = data.drop(params["objective_variable"], axis=1)

    # Out of Sample Predictions
    y_out_of_sample = mmm.sample_posterior_predictive(
        X_pred=data_test, extend_idata=False
    )
    y_out_of_sample_with_adstock = mmm.sample_posterior_predictive(
        X_pred=data_test, extend_idata=False, include_last_observations=True
    )

    # Plot Funcs
    def plot_in_sample(X, y, ax, n_points: int = 15):
        (
            y.to_frame()
            .set_index(X[params["date_column"]])
            .iloc[-n_points:]
            .plot(ax=ax, color="black", label="actuals")
        )

    def plot_out_of_sample(X_out_of_sample, y_out_of_sample, ax, color, label):
        print(X_out_of_sample[params["date_column"]][0])
        X_out_of_sample[params["date_column"]] = pd.to_datetime(
            X_out_of_sample[params["date_column"]]
        )
        print(type(X_out_of_sample[params["date_column"]][0]))
        y_out_of_sample_groupby = y_out_of_sample["y"].to_series().groupby("date")

        lower, upper = quantiles = [0.025, 0.975]
        conf = y_out_of_sample_groupby.quantile(quantiles).unstack()
        ax.fill_between(
            X_out_of_sample[params["date_column"]],  # .dt.to_pydatetime(),
            conf[lower],
            conf[upper],
            alpha=0.25,
            color=color,
            label=f"{label} interval",
        )

        mean = y_out_of_sample_groupby.mean()
        mean.plot(ax=ax, label=label, color=color, linestyle="--")
        ax.set(
            ylabel="Original Target Scale", title="Out of sample predictions for MMM"
        )

        return ax

    # Plots
    _, ax = plt.subplots()
    plot_in_sample(X, y, ax=ax)
    plot_out_of_sample(
        X_out_of_sample, y_out_of_sample, ax=ax, label="out of sample", color="C0"
    )
    plot_out_of_sample(
        X_out_of_sample,
        y_out_of_sample_with_adstock,
        ax=ax,
        label="adstock out of sample",
        color="C1",
    )

    return ax.legend()
