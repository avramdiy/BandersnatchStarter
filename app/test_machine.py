import os
import pandas as pd
from .machine import Machine  # Use relative import

# Sample DataFrame for testing
data = pd.DataFrame({
    "Energy": [0.5, 0.6, 0.7],
    "Health": [0.4, 0.6, 0.8],
    "Level": [1, 2, 3],
    "Sanity": [0.3, 0.5, 0.7],
    "Rarity": ["common", "rare", "epic"]
})

model_name = "test_model"
model_path = os.path.join(Machine.model_directory, f"{model_name}.joblib")

def test_save():
    # Instantiate Machine and save model
    machine_instance = Machine(data, target_column="Rarity", model_name=model_name)
    
    # Check if the model file was created
    if os.path.exists(model_path):
        print(f"[SUCCESS] Model saved successfully at {model_path}.")
    else:
        print(f"[FAIL] Model was not saved at {model_path}.")

def test_open():
    # Load the saved model
    loaded_machine = Machine.open(model_name)
    
    # Check if the loaded instance is a Machine object with a model
    if isinstance(loaded_machine, Machine) and hasattr(loaded_machine, 'model'):
        print("[SUCCESS] Model loaded successfully with open().")
    else:
        print("[FAIL] Model could not be loaded with open().")

def test_info():
    # Create a new Machine instance to test info()
    machine_instance = Machine(data, target_column="Rarity")
    info_str = machine_instance.info()

    # Check if model info contains base model name and timestamp
    if "Base Model: RandomForestClassifier" in info_str and "Timestamp:" in info_str:
        print(f"[SUCCESS] info() returned the correct model name and timestamp:\n{info_str}")
    else:
        print(f"[FAIL] info() did not return the expected information:\n{info_str}")

# Run the tests
test_save()
test_open()
test_info()
