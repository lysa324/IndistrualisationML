"""
This is a boilerplate pipeline 'SplitData'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import splitData

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

	node(
		func = splitData,
		inputs ="shaped_datas",
		outputs = ["shaped_datas_train","shaped_datas_train_label","shaped_datas_test","shaped_datas_test_label"],
		name =  "split_data_node"
	)
])






