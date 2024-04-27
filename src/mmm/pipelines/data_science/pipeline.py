"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    channel_contributions,
    channel_roas,
    model_diagnostics,
    model_training,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                model_training,
                inputs=["feature_data", "params:model_specification"],
                outputs="fitted_model",
                name="model_training",
            ),
            node(
                model_diagnostics,
                inputs=["fitted_model", "params:model_specification"],
                outputs=[
                    "model_summary",
                    "model_trace",
                    "model_posterior_predictive",
                    "model_components_contributions",
                    "model_contribution_breakdown",
                ],
                name="model_diagnostics",
            ),
            node(
                channel_contributions,
                inputs="fitted_model",
                outputs=[
                    "channel_contributions",
                    "channel_alphas",
                    "channel_lam",
                    "channel_contribution",
                    "channel_direct_contribution",
                    "channel_contribution_func",
                    "channel_contribution_func_abs",
                ],
                name="channel_contributions",
            ),
            node(
                channel_roas,
                inputs=["feature_data", "fitted_model", "params:model_specification"],
                outputs="channel_roas",
                name="channel_roas",
            ),
        ]
    )
