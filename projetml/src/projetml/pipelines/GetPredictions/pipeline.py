"""
This is a boilerplate pipeline 'GetPredictions'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import predict



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
	node(
		func = predict,
		inputs = ["shaped_data_GetPredictions_API"],
		outputs = "predicted_classes_API",
		name =   "GetPredictions"
	)
])

