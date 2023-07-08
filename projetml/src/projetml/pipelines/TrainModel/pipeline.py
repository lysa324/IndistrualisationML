"""
This is a boilerplate pipeline 'TrainModel'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline


from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "shaped_datas_train",
                    "shaped_datas_train",
                    "shaped_datas_train_label",
                    "shaped_datas_test",
                    "shaped_datas_test_label",
                ],
                outputs="shaped_model",
                name="train_model_node",
            )
        ]
    )
