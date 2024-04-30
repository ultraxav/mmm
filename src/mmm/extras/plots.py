from typing import Any, Dict

import arviz as az
import pandas as pd
import plotly.graph_objs as go
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM


def posterior_predictive_check_plot(
    mmm: DelayedSaturatedMMM, params: Dict[str, Any]
) -> go.Figure:
    likelihood_hdi_94 = az.hdi(ary=mmm.idata["posterior_predictive"], hdi_prob=0.94)[
        "y"
    ]
    likelihood_hdi_50 = az.hdi(ary=mmm.idata["posterior_predictive"], hdi_prob=0.50)[
        "y"
    ]

    likelihood_hdi_94 = mmm.get_target_transformer().inverse_transform(
        Xt=likelihood_hdi_94
    )
    likelihood_hdi_50 = mmm.get_target_transformer().inverse_transform(
        Xt=likelihood_hdi_50
    )

    data = {
        "date_week": mmm.idata["fit_data"][params["date_column"]].to_numpy(),
        "revenue": mmm.idata["fit_data"]["y"].to_numpy(),
        "likelihood_hdi_94_0": likelihood_hdi_94[:, 0],
        "likelihood_hdi_94_1": likelihood_hdi_94[:, 1],
        "likelihood_hdi_50_0": likelihood_hdi_50[:, 0],
        "likelihood_hdi_50_1": likelihood_hdi_50[:, 1],
    }

    data = pd.DataFrame(data)
    data["date_week"] = pd.to_datetime(data["date_week"])

    # Create a Plotly figure
    fig = go.Figure()

    color = "cornflowerblue"

    # HDI 94
    fig.add_trace(
        go.Scatter(
            x=data[params["date_column"]],
            y=data["likelihood_hdi_94_0"],
            mode="lines",
            name="Lower HDI 94",
            line=dict(color=color),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[params["date_column"]],
            y=data["likelihood_hdi_94_1"],
            fill="tonexty",
            mode="lines",
            name="Upper HDI 94",
            line=dict(color=color),
        )
    )

    # HDI 50
    fig.add_trace(
        go.Scatter(
            x=data[params["date_column"]],
            y=data["likelihood_hdi_50_0"],
            mode="lines",
            name="Lower HDI 50",
            line=dict(color=color),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[params["date_column"]],
            y=data["likelihood_hdi_50_1"],
            fill="tonexty",
            mode="lines",
            name="Upper HDI 50",
            line=dict(color=color),
        )
    )

    # Add the price chart
    fig.add_trace(
        go.Scatter(
            x=data[params["date_column"]],
            y=data[params["objective_variable"]],
            name=params["objective_variable"],
            line=dict(color="black"),
        )
    )

    fig.update_layout(
        title="Posterior Predictive Check",
        xaxis_title=params["date_column"],
        yaxis_title=params["objective_variable"],
    )

    return fig


def contribution_breakdown_over_time_plot(
    mmm: DelayedSaturatedMMM, original_scale=True
) -> go.Figure:
    data = mmm.compute_mean_contributions_over_time(original_scale=original_scale)

    data["total"] = 1

    fig = go.Figure()
    for channel in data.columns[::-1][1:]:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[channel],
                hoverinfo="x+y",
                mode="lines",
                line=dict(width=0.5),
                stackgroup="one",  # define stack group
                name=channel,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["total"],
            hoverinfo="x+y",
            mode="lines",
            line=dict(width=2, color="black"),
            stackgroup="one",  # define stack group
            name="total",
        )
    )

    return fig
