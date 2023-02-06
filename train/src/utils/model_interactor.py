import sys
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel
from torch import nn

sys.path.append("..")
from train.src.models.convlstm.seq2seq import Seq2Seq  # noqa: E402
from train.src.models.obpoint_seq2seq.obpoint_seq2seq import OBPointSeq2Seq  # noqa: E402
from train.src.models.self_attention_convlstm.sa_seq2seq import SASeq2Seq  # noqa: E402
from train.src.models.self_attention_memory_convlstm.sam_seq2seq import SAMSeq2Seq  # noqa: E402
from train.src.models.test_model.test_model import TestModel  # noqa: E402


class ModelInteractor:
    """Initialzie and return the model by a given model name."""

    def __init__(self) -> None:
        super(ModelInteractor, self).__init__()

    def initialize_model(self, model_name: str, **kwargs) -> Any:
        params = ModelParams(**kwargs)
        if model_name == ModelName.Seq2Seq:
            return Seq2Seq(
                params.num_channels,
                params.kernel_size,
                params.num_kernels,
                params.padding,
                params.activation,
                params.frame_size,
                params.num_layers,
                params.input_seq_length,
                params.out_channels,
                params.weights_initializer,
                params.return_sequences,
            )
        elif model_name == ModelName.SASeq2Seq:
            if not isinstance(params.attention_hidden_dims, int):
                raise ValueError(f"attention_hidden_dims=[int] shoulb be passed when i{model_name} is initialied.")

            return SASeq2Seq(
                params.attention_hidden_dims,
                params.num_channels,
                params.kernel_size,
                params.num_kernels,
                params.padding,
                params.activation,
                params.frame_size,
                params.num_layers,
                params.input_seq_length,
                params.out_channels,
                params.weights_initializer,
                params.return_sequences,
            )
        elif model_name == ModelName.SAMSeq2Seq:
            if not isinstance(params.attention_hidden_dims, int):
                raise ValueError(f"attention_hidden_dims=[int] shoulb be passed when {model_name} is initialied.")

            return SAMSeq2Seq(
                params.attention_hidden_dims,
                params.num_channels,
                params.kernel_size,
                params.num_kernels,
                params.padding,
                params.activation,
                params.frame_size,
                params.num_layers,
                params.input_seq_length,
                params.out_channels,
                params.weights_initializer,
                params.return_sequences,
            )

        elif model_name == ModelName.OBPointSeq2Seq:
            if not isinstance(params.ob_point_count, int):
                raise ValueError(f"ob_point_count: [int] shoulb be passed when {model_name} is initialied.")

            if not isinstance(params.prediction_seq_length, int):
                raise ValueError(f"prediction_seq_length: [int] shoulb be passed when {model_name} is initialied.")

            return OBPointSeq2Seq(
                params.num_channels,
                params.ob_point_count,
                params.kernel_size,
                params.num_kernels,
                params.padding,
                params.activation,
                params.frame_size,
                params.num_layers,
                params.input_seq_length,
                params.prediction_seq_length,
                params.out_channels,
                params.weights_initializer,
                params.return_sequences,
            )
        elif model_name == ModelName.TestModel:
            return TestModel(params.return_sequences)
        else:
            raise ValueError(f"Unknown model {model_name}")

    def save_model(self, model: nn.Module, save_path: str) -> None:
        if isinstance(model, Seq2Seq):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_channels": model.num_channels,
                    "kernel_size": model.kernel_size,
                    "num_kernels": model.num_kernels,
                    "padding": model.padding,
                    "activation": model.activation,
                    "frame_size": model.frame_size,
                    "num_layers": model.num_layers,
                    "input_seq_length": model.input_seq_length,
                    "out_channels": model.out_channels,
                    "weights_initializer": model.weights_initializer,
                    "return_sequences": model.return_sequences,
                },
                save_path,
            )
        elif isinstance(model, SASeq2Seq) or isinstance(model, SAMSeq2Seq):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "attention_hidden_dims": model.attention_hidden_dims,
                    "num_channels": model.num_channels,
                    "kernel_size": model.kernel_size,
                    "num_kernels": model.num_kernels,
                    "padding": model.padding,
                    "activation": model.activation,
                    "frame_size": model.frame_size,
                    "num_layers": model.num_layers,
                    "input_seq_length": model.input_seq_length,
                    "out_channels": model.out_channels,
                    "weights_initializer": model.weights_initializer,
                    "return_sequences": model.return_sequences,
                },
                save_path,
            )
        elif isinstance(model, OBPointSeq2Seq):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_channels": model.num_channels,
                    "ob_point_count": model.ob_point_count,
                    "kernel_size": model.kernel_size,
                    "num_kernels": model.num_kernels,
                    "padding": model.padding,
                    "activation": model.activation,
                    "frame_size": model.frame_size,
                    "num_layers": model.num_layers,
                    "input_seq_length": model.input_seq_length,
                    "prediction_seq_length": model.prediction_seq_length,
                    "out_channels": model.out_channels,
                    "weights_initializer": model.weights_initializer,
                    "return_sequences": model.return_sequences,
                },
                save_path,
            )
        elif isinstance(model, TestModel):
            torch.save({"model_state_dict": model.state_dict()}, save_path)
        else:
            raise ValueError(f"Unknown model {model.__class__.__name__}")


class ModelParams(BaseModel):
    """Interface of model params"""

    # Common parameters
    num_channels: int
    kernel_size: Union[int, Tuple]
    num_kernels: int
    padding: Union[int, Tuple, str]
    activation: str
    frame_size: Tuple
    num_layers: int
    input_seq_length: int
    out_channels: Optional[int] = None
    weights_initializer: str
    return_sequences: bool

    # Only needed for SASeq2Seq or SAMSeq2Seq
    attention_hidden_dims: Optional[int] = None

    # Only needed for OBPointSeq2Seq
    ob_point_count: Optional[int] = None
    prediction_seq_length: Optional[int] = None


class ModelName(str, Enum):
    """Enum class of model names."""

    Seq2Seq = "Seq2Seq"
    SASeq2Seq = "SASeq2Seq"
    SAMSeq2Seq = "SAMSeq2Seq"
    OBPointSeq2Seq = "OBPointSeq2Seq"
    TestModel = "TestModel"

    @staticmethod
    def all_names() -> List[str]:
        return [v.value for v in ModelName.__members__.values()]

    @staticmethod
    def is_valid(value: str) -> bool:
        return value in ModelName.all_names()
