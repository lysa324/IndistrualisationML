"""
This is a boilerplate pipeline 'PreprocessData'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import datapreprocess


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=datapreprocess,
                inputs="raw_daily_data",
                outputs="shaped_datas",
                name="node_merge_raw_daily_data",
            )
        ]
    )
