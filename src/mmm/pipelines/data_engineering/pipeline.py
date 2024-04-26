"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_cleaning, feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                data_cleaning,
                inputs="raw_data",
                outputs="primary_data",
                name="data_cleaning",
            ),
            node(
                feature_engineering,
                inputs="primary_data",
                outputs=["feature_data", "test_data"],
                name="feature_engineering",
            ),
        ]
    )
