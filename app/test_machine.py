import unittest
import os
import pandas as pd  # Import pandas if using DataFrame in tests
from app.machine import Machine  # Adjust the import based on your directory structure
from datetime import datetime

class TestMachine(unittest.TestCase):
    def setUp(self):
        # Set the model path to the pre-existing model
        self.model_path = r"C:\Users\Ev\Desktop\Bandersnatch\models\model.joblib"
        
        # Print the expected model path
        print("Expected model path:", self.model_path)

        # Load the pre-existing model
        if os.path.exists(self.model_path):
            self.machine = Machine.open("model")  # Load the model without the .joblib extension
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def test_info(self):
        info_str = self.machine.info()
        timestamp_str = info_str.split("Timestamp: ")[1].split("\n")[0]
        self.assertIsNotNone(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'), "Timestamp should be in the correct format.")

    def test_predict(self):
        # Example features for prediction (adjust as needed for your dataset)
        test_features = {
            "feature1": [1.0],  # Replace with actual feature names and values
            "feature2": [0.5],
            # Add all required features here
        }
        features_df = pd.DataFrame(test_features)
        prediction, probability = self.machine.predict(features_df)
        self.assertIsNotNone(prediction)
        self.assertGreaterEqual(probability, 0)

    def test_save(self):
        # Test if the model saves correctly to the specified path
        test_model_name = "test_model"
        self.machine.save(test_model_name)  # Save as a new model for testing
        saved_model_path = os.path.join(r'C:\Users\Ev\Desktop\Bandersnatch\models', f"{test_model_name}.joblib")
        self.assertTrue(os.path.exists(saved_model_path), "Model file should exist after saving.")

if __name__ == "__main__":
    unittest.main()
