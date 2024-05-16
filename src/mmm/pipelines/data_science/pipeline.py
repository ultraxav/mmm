"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    channel_contributions,
    model_diagnostics,
    model_training,
    out_of_sample_preds
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
                    "model_summary_plot",
                ],
                name="model_diagnostics",
            ),
            node(
                channel_contributions,
                inputs=["feature_data", "fitted_model", "params:model_specification"],
                outputs=[
                    "channel_contribution_summary",
                    "channel_contribution_breakdown",
                    "channel_alpha",
                    "channel_lam",
                    "channel_beta",
                    "channel_contribution",
                    "channel_direct_contribution",
                    "channel_roas",
                ],
                name="channel_contributions",
            ),
            node(
                out_of_sample_preds,
                inputs=[
                    "feature_data",
                    "test_data",
                    "fitted_model",
                    "params:model_specification",
                ],
                outputs="out_of_sample_preds_plot",
                name="out_of_sample_preds",
            ),
        ]
    )
