import unittest
import cv2
import numpy as np
from numpy.testing import assert_array_equal
from pprint import pprint
from annotation.image_utils import ten_crop, normalise_single_labels, normalise_multiple_labels

class TestTenCropLabeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Preprocess some data for testing
        TEST_IMAGE = "./images/test_images/fl.png"
        image = cv2.imread(TEST_IMAGE)
        cls.elbow_inputs = [(0,0), (500, 32), (64, 484), (484, 500), (258, 258)]
        cls.wrist_inputs = [(500,32), (0, 0), (64, 484), (484, 500)]
        cls.preprocessed_data = ten_crop(image, cls.elbow_inputs, cls.wrist_inputs)

    def test_single_label(self):
        true_values = [
            [1, 0, 0, 1, 0, 0],                  # Top-left crop
            [1, 0.998, 0, 1, 0.998, 0],          # Flipped top-left crop

            [1, 0.154, 0.923, 1, 0.154, 0.923],  # Bottom-left crop          
            [1, 0.844, 0.923, 1, 0.844, 0.923],  # Flipped bottom-left crop

            [1, 0.962, 0.077, 1, 0.962, 0.077],  # Top-right crop
            [1, 0.036, 0.077, 1, 0.036, 0.077],  # Flipped top-right crop

            [1, 0.923, 0.962, 1, 0.923, 0.962],  # Bottom-right crop
            [1, 0.075, 0.962, 1, 0.075, 0.962],  # Flipped bottom-right crop

            [1, 0.5, 0.5, 0, 0, 0],              # Center crop
            [1, 0.498, 0.5, 0, 0, 0],            # Flipped center crop
        ]

        for crop, true_value in zip(self.preprocessed_data, true_values):
            actual_value = normalise_single_labels(crop["elbow_pos"], crop["wrist_pos"])
            self.assertEqual(actual_value, true_value)
    
    def test_multi_label(self):
        true_values = [np.zeros((13, 13, 6)) for _ in range(10)]

        # Top-left crop
        # elbow, wrist inputs: (0,0), (0,0) - effective positions (0,0)
        box_x, box_y = 0, 0
        values = [1, 0, 0, 1, 0, 0]
        true_values[0][box_x, box_y, :] = values
        # elbow input: (258, 258) - effective position (258,  258)
        box_x, box_y = 8, 8
        values = [1, 0.062, 0.062, 0, 0, 0]
        true_values[0][box_x, box_y, :] = values

        # Flipped top-left crop
        # elbow, wrist inputs: (0,0), (0,0) - effective positions (415,0), (415,0)
        box_x, box_y = 12, 0
        values = [1, 0.969, 0, 1, 0.969, 0]
        true_values[1][box_x, box_y, :] = values
        # elbow input: (258, 258) - effective position (157, 258)
        box_x, box_y = 4, 8
        values = [1, 0.906, 0.062, 0, 0, 0]
        true_values[1][box_x, box_y, :] = values

        # Bottom-left crop
        # elbow, wrist inputs: (64,484), (64,484) - effective positions (64, 384), (64, 384)
        box_x, box_y = 2, 12
        values = [1, 0, 0, 1, 0, 0]
        true_values[2][box_x, box_y, :] = values
        # elbow input: (258, 258) - effective positon (258, 158)
        box_x, box_y = 8, 4
        values = [1, 0.062, 0.938, 0, 0, 0]
        true_values[2][box_x, box_y, :] = values

        # Flipped bottom-left crop
        # elbow, wrist inputs: (64,484), (64,484) - effective positions (351, 384), (351, 384)
        box_x, box_y = 10, 12
        values = [1, 0.969, 0, 1, 0.969, 0]
        true_values[3][box_x, box_y, :] = values
        # elbow: (258, 258) - effective position (157, 158)
        box_x, box_y = 4, 4
        values = [1, 0.906, 0.938, 0, 0, 0]
        true_values[3][box_x, box_y, :] = values

        # Top-right crop
        # elbow, wrist inputs: (500,32), (500,32) - effective positions (400, 32), (400, 32)
        box_x, box_y = 12, 1
        values = [1, 0.5, 0, 1, 0.5, 0]
        true_values[4][box_x, box_y, :] = values
        # elbow input: (258, 258) - effective positon (158, 258)
        box_x, box_y = 4, 8
        values = [1, 0.938, 0.062, 0, 0, 0]
        true_values[4][box_x, box_y, :] = values

        # Flipped top-right crop
        # elbow, wrist inputs: (500,32), (500,32) - effective positions (15, 32), (15, 32)
        box_x, box_y = 0, 1
        values = [1, 0.469, 0, 1, 0.469, 0]
        true_values[5][box_x, box_y, :] = values
        # elbow: (258, 258) - effective position (257, 258)
        box_x, box_y = 8, 8
        values = [1, 0.031, 0.062, 0, 0, 0]
        true_values[5][box_x, box_y, :] = values

        # Bottom-right crop
        # elbow, wrist inputs: (484,500), (484,500) - effective positions (384, 400), (384, 300)
        box_x, box_y = 12, 12
        values = [1, 0, 0.5, 1, 0, 0.5]
        true_values[6][box_x, box_y, :] = values
        # elbow input: (258, 258) - effective positon (158, 158)
        box_x, box_y = 4, 4
        values = [1, 0.938, 0.938, 0, 0, 0]
        true_values[6][box_x, box_y, :] = values

        # Flipped bottom-right crop
        # elbow, wrist inputs: (500,32), (500,32) - effective positions (31, 400), (31, 400)
        box_x, box_y = 0, 12
        values = [1, 0.969, 0.5, 1, 0.969, 0.5]
        true_values[7][box_x, box_y, :] = values
        # elbow: (258, 258) - effective position (257, 158)
        box_x, box_y = 8, 4
        values = [1, 0.031, 0.938, 0, 0, 0]
        true_values[7][box_x, box_y, :] = values

        # Center crop
        # elbow input: (258, 258) - effective positon (208, 208)
        box_x, box_y = 6, 6
        values = [1, 0.5, 0.5, 0, 0, 0]
        true_values[8][box_x, box_y, :] = values

        # Flipped center crop
        # elbow: (258, 258) - effective position (207, 208)
        box_x, box_y = 6, 6
        values = [1, 0.469, 0.5, 0, 0, 0]
        true_values[9][box_x, box_y, :] = values

        for crop, true_value in zip(self.preprocessed_data, true_values):
            actual_value = normalise_multiple_labels(crop["elbow_pos"], crop["wrist_pos"])
    
            if not np.array_equal(actual_value, true_value):
                # Find the indices where the arrays differ
                diff_indices = np.where(actual_value != true_value)
                print("Arrays differ at indices:")
                pprint(diff_indices)
                # Print the actual and expected values at the differing indices
                print("Actual values:")
                print(actual_value[diff_indices])
                print("Expected values:")
                print(true_value[diff_indices])
            
            # Assert that the arrays are equal for validation
            assert_array_equal(actual_value, true_value)

if __name__ == '__main__':
    unittest.main()