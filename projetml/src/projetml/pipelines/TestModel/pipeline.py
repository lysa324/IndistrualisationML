"""
This is a boilerplate pipeline 'TestModel'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import test_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=test_model,
                inputs=["shaped_model", "shaped_data_predictions"],
                outputs="predicted_classes",
                name="test_model_node",
            )
        ]
    )
