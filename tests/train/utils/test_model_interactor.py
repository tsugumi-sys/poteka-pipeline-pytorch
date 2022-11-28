import unittest
import sys

sys.path.append(".")
from train.src.utils.model_interactor import ModelInteractor, ModelName
from train.src.models.convlstm.seq2seq import Seq2Seq
from train.src.models.self_attention_convlstm.sa_seq2seq import SASeq2Seq
from train.src.models.self_attention_memory_convlstm.sam_seq2seq import SAMSeq2Seq
from train.src.models.obpoint_seq2seq.obpoint_seq2seq import OBPointSeq2Seq
from train.src.models.test_model.test_model import TestModel


class TestModelInteractor(unittest.TestCase):
    def test_initialize_model(self):
        num_channels = 3
        kernel_size = 3
        num_kernels = 16
        padding = "same"
        activation = "relu"
        frame_size = (50, 50)
        num_layers = 3
        input_seq_length = 6
        return_sequences = False
        weights_initializer = "he"
        attention_hidden_dims = 4
        ob_point_count = 35
        label_seq_length = 6

        interactor = ModelInteractor()

        model = interactor.initialize_model(
            "Seq2Seq",
            num_channels=num_channels,
            kernel_size=kernel_size,
            num_kernels=num_kernels,
            padding=padding,
            activation=activation,
            frame_size=frame_size,
            num_layers=num_layers,
            input_seq_length=input_seq_length,
            out_channels=None if return_sequences is False else 1,
            weights_initializer=weights_initializer,
            return_sequences=return_sequences,
            attention_hidden_dims=attention_hidden_dims,
            ob_point_count=ob_point_count,
            prediction_seq_length=label_seq_length,
        )
        self.assertIsInstance(model, Seq2Seq)

        model = interactor.initialize_model(
            "SASeq2Seq",
            num_channels=num_channels,
            kernel_size=kernel_size,
            num_kernels=num_kernels,
            padding=padding,
            activation=activation,
            frame_size=frame_size,
            num_layers=num_layers,
            input_seq_length=input_seq_length,
            out_channels=None if return_sequences is False else 1,
            weights_initializer=weights_initializer,
            return_sequences=return_sequences,
            attention_hidden_dims=attention_hidden_dims,
            ob_point_count=ob_point_count,
            prediction_seq_length=label_seq_length,
        )
        self.assertIsInstance(model, SASeq2Seq)

        model = interactor.initialize_model(
            "SAMSeq2Seq",
            num_channels=num_channels,
            kernel_size=kernel_size,
            num_kernels=num_kernels,
            padding=padding,
            activation=activation,
            frame_size=frame_size,
            num_layers=num_layers,
            input_seq_length=input_seq_length,
            out_channels=None if return_sequences is False else 1,
            weights_initializer=weights_initializer,
            return_sequences=return_sequences,
            attention_hidden_dims=attention_hidden_dims,
            ob_point_count=ob_point_count,
            prediction_seq_length=label_seq_length,
        )
        self.assertIsInstance(model, SAMSeq2Seq)

        model = interactor.initialize_model(
            "OBPointSeq2Seq",
            num_channels=num_channels,
            kernel_size=kernel_size,
            num_kernels=num_kernels,
            padding=padding,
            activation=activation,
            frame_size=frame_size,
            num_layers=num_layers,
            input_seq_length=input_seq_length,
            out_channels=None if return_sequences is False else 1,
            weights_initializer=weights_initializer,
            return_sequences=return_sequences,
            attention_hidden_dims=attention_hidden_dims,
            ob_point_count=ob_point_count,
            prediction_seq_length=label_seq_length,
        )
        self.assertIsInstance(model, OBPointSeq2Seq)

        model = interactor.initialize_model(
            "TestModel",
            num_channels=num_channels,
            kernel_size=kernel_size,
            num_kernels=num_kernels,
            padding=padding,
            activation=activation,
            frame_size=frame_size,
            num_layers=num_layers,
            input_seq_length=input_seq_length,
            out_channels=None if return_sequences is False else 1,
            weights_initializer=weights_initializer,
            return_sequences=return_sequences,
            attention_hidden_dims=attention_hidden_dims,
            ob_point_count=ob_point_count,
            prediction_seq_length=label_seq_length,
        )
        self.assertIsInstance(model, TestModel)

        with self.assertRaises(ValueError):
            _ = interactor.initialize_model("UnknownModel")
