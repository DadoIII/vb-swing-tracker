import unittest
import cv2
from src.image_utils import ten_crop, normalise_single_labels, normalise_multiple_labels

class TestTenCropLabeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Preprocess some data for testing
        TEST_IMAGE = "./test_images/fl.png"
        image = cv2.imread(TEST_IMAGE)
        cls.preprocessed_data = ten_crop(image, [(0,0), (500, 32), (64, 484), (484, 500), (258, 258)], [(500,32), (0, 0), (64, 484), (484, 500)])

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
            self.assertEqual(normalise_single_labels(crop["elbow_pos"], crop["wrist_pos"]), true_value)
    
    def test_multi_label(self):
        pass

if __name__ == '__main__':
    unittest.main()