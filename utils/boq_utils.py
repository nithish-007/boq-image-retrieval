import yaml
import os
import torch
import numpy as np
import coremltools as ct
import logging
from torch.export import export
from executorch import exir
from executorch.exir import ExecutorchBackendConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower
from executorch.exir.backend.backend_api import to_backend

logger = logging.getLogger("BoQ_train")

def export_model_to_pte(model, example_input, output_file: str):
    try:
        os.makedirs("weights", exist_ok=True)
        model.eval()
        exported_program = export(model, example_input)
        edge = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
        )
        exec_prog = edge.to_executorch()
        with open(output_file, "wb") as file:
            exec_prog.write_to_file(file)
        return output_file
        
    except Exception as e:
        raise FileNotFoundError("The Model export failed")

def check_directory_empty(directory_path):
    if os.path.isdir(directory_path) and os.listdir(directory_path):
        raise FileExistsError(f"The directory {directory_path} is not empty.")

def load_yaml_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def convert_to_coreml(model, example_input, coreml_path):
    # Convert the PyTorch Lightning model to a TorchScript model using tracing
    try:
        traced_model = model.to_torchscript(method='trace', example_inputs=example_input)
        ml_model = ct.convert(
            traced_model,
            convert_to="neuralnetwork",
            inputs=[ct.TensorType(shape=example_input.shape)]
        )
        ml_model.save(coreml_path)
    except Exception as e:
        logger.error(f"Model successfully exported to Core ML at {coreml_path}")

if __name__ == '__main__':
    pass
