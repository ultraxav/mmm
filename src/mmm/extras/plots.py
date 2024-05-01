from typing import Any, Dict

import arviz as az
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
from sklearn.metrics import mean_absolute_percentage_error, r2_score


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
        "date_week": mmm.idata["fit_data"]["date_week"].to_numpy(),
        f'{params["objective_variable"]}_actuals': mmm.idata["fit_data"][
            "y"
        ].to_numpy(),
        f'{params["objective_variable"]}_predicted': mmm.compute_mean_contributions_over_time(
            original_scale=True
        )
        .sum(axis=1)
        .to_numpy(),
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

    # KPI
    fig.add_trace(
        go.Scatter(
            x=data[params["date_column"]],
            y=data[f'{params["objective_variable"]}_actuals'],
            name=f'{params["objective_variable"]}_actuals',
            line=dict(color="black"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["date_week"],
            y=data[f'{params["objective_variable"]}_predicted'],
            name=f'{params["objective_variable"]}_predicted',
            line=dict(color="red"),
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

    fig.update_layout(
        title="Contribution Breakdown over Time",
    )

    return fig


def model_summary_plot(mmm: DelayedSaturatedMMM) -> go.Figure:
    data = {
        "y_actuals": mmm.idata["fit_data"]["y"].to_numpy(),
        "y_predicted": mmm.compute_mean_contributions_over_time(original_scale=True)
        .sum(axis=1)
        .to_numpy(),
    }
    data["residuals"] = data["y_actuals"] - data["y_predicted"]

    r2 = r2_score(data["y_actuals"], data["y_predicted"])
    mape = mean_absolute_percentage_error(data["y_actuals"], data["y_predicted"])

    fig = make_subplots(rows=4, cols=1)

    fig.add_trace(
        go.Scatter(
            x=data["y_predicted"],
            y=data["y_actuals"],
            mode="markers",
            name="Predicted vs actuals",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["y_predicted"],
            y=data["y_predicted"],
            mode="lines",
            name="Fitted Prediction",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(x=data["residuals"], name="Residuals Distribution"), row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(data["y_predicted"]), 1)),
            y=data["y_predicted"],
            mode="lines",
            name="Predicted",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(data["y_actuals"]), 1)),
            y=data["y_actuals"],
            mode="lines",
            name="Actuals",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(data["residuals"]), 1)),
            y=data["residuals"],
            name="Residuals",
        ),
        row=4,
        col=1,
    )

    # fig.update_layout(height=1000, width=1500, title_text="Model Summary Plot")

    fig.update_layout(
        title=f"Model Summary - R2 Score: {round(r2, 4)} - MAPE: {round(mape, 4)}",
    )

    return fig
