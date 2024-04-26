"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_diagnostics, model_training


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
                inputs="fitted_model",
                outputs=[
                    "model_summary",
                    "model_trace",
                    "model_posterior_predictive",
                    "model_components_contributions",
                    "model_contribution_breakdown",
                    "model_contributions",
                ],
                name="model_diagnostics",
            ),
        ]
    )
