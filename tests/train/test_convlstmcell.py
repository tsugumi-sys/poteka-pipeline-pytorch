import unittest

import torch

from train.src.config import DEVICE, WeightsInitializer
from train.src.convlstmcell import ConvLSTMCell


class TestConvLSTMCell(unittest.TestCase):
    def test_convlstmcell(self):
        model = (
            ConvLSTMCell(
                in_channels=3,
                out_channels=3,
                kernel_size=(3, 3),
                padding=1,
                activation="relu",
                frame_size=(50, 50),
                weights_initializer=WeightsInitializer.Zeros,
            )
            .to(DEVICE)
            .to(torch.float)
        )
        self.assertEqual(model.conv.weight.dtype, torch.float)


if __name__ == "__main__":
    unittest.main()
