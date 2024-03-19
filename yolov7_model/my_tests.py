import unittest
import torch
import numpy as np

from keypoint_finetune import CustomLoss, CustomDataset

class TestTenCropLabeling(unittest.TestCase):
    custom_loss = CustomLoss()

    def test_single_label(self):
        tests = [
            (torch.Tensor([10, 10, 1, 25, 25, 1]).view(1,1,1,6),
             torch.Tensor([10, 10, 1, 25, 25, 1]).view(1,1,1,6),  # Same tensors with detection
             0),
            (torch.Tensor([10, 10, 0, 25, 25, 0]).view(1,1,1,6),
             torch.Tensor([10, 10, 0, 25, 25, 0]).view(1,1,1,6),  # Same tensors without detection
             0),
            (torch.Tensor([100, 100, 0, 250, 250, 0]).view(1,1,1,6),
             torch.Tensor([10, 10, 0, 25, 25, 0]).view(1,1,1,6),  # No detection different positions
             0),
            (torch.Tensor([10, 10, 0, 25, 25, 1]).view(1,1,1,6),
             torch.Tensor([10, 10, 1, 25, 25, 1]).view(1,1,1,6),  # Different confidence
             100),
            (torch.Tensor([10, 10, 1, 25, 25, 0]).view(1,1,1,6),
             torch.Tensor([10, 10, 1, 25, 25, 1]).view(1,1,1,6),  # Different confidence
             100),
            (torch.Tensor([100, 100, 1, 250, 250, 1]).view(1,1,1,6),
             torch.Tensor([10, 10, 0, 25, 25, 0]).view(1,1,1,6),  # No detection, different confidence and positions
             200),
            (torch.Tensor([15, 15, 1, 30, 30, 1]).view(1,1,1,6),
             torch.Tensor([10, 10, 1, 25, 25, 1]).view(1,1,1,6),  # Detection with different positions
             4 * 25),
            (torch.Tensor([15, 15, 0, 30, 30, 0]).view(1,1,1,6),
             torch.Tensor([10, 10, 1, 25, 25, 1]).view(1,1,1,6),  # Detection with different confidence and positions
             4 * 25 + 2 * 100),
            (torch.Tensor([15, 15, 0, 30, 30, 0] * 8).view(2,2,2,6),
             torch.Tensor([10, 10, 1, 25, 25, 1] * 8).view(2,2,2,6),  # Multiple batches and grid cells
             4 * 25 + 2 * 100),
        ]

        for pred, true, loss in tests:
            try:
                computed_loss = self.custom_loss.compute_scale_loss(pred, true)
                self.assertEqual(computed_loss, loss)
            except AssertionError:
                print(f"Failed for pred: {pred}, true: {true}, expected_loss: {loss}, computed_loss: {computed_loss}")
                raise  # Re-raise the AssertionError to stop the test

    def test_get_targets(self):
        custom_dataset = CustomDataset([], [{"elbows_right": [(0.5, 0.5), (0.2, 0.1)], 
                                             "wrists_right": [], 
                                             "elbows_left":  [(0.2, 0.1)], 
                                             "wrists_left":  [(0.15, 0.25)]}])


        true_target = np.zeros((10, 10, 12))
        true_target[5,5,:3] = [0,0,1]
        true_target[2,1,:3] = [0,0,1]
        true_target[2,1,6:9] = [0,0,1]
        true_target[1,2,9:] = [0.5,0.5,1]

        try:
            self.assertTrue(np.array_equal(custom_dataset.get_target(0, 10, 10), true_target))
        except AssertionError:
            np.set_printoptions(threshold=10_000)
            print(f"Test failed predicted target:\n{custom_dataset.get_target(0, 10, 10)}\nTrue target:\n{true_target}")
            raise  # Re-raise the AssertionError to stop the test


if __name__ == '__main__':
    unittest.main()