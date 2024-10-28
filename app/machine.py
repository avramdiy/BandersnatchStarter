import os
import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Tuple, List
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Machine:
    """
    A Machine class for managing a RandomForestClassifier model,
    training it, saving it, and making predictions with probabilities.
    """

    models = []  # Class-level attribute to store instances of Machine
    model_directory = 'C:\\Users\\Ev\\Desktop\\Bandersnatch\\models'  # Path to store models

    def __init__(self, data: DataFrame, target_column: str = "Rarity", n_estimators: int = 100, model_name: str = None):
        """
        Initializes the Machine with a RandomForestClassifier model, sets up feature
        and target data, and trains the model.

        :param data: DataFrame containing the dataset, including features and target.
        :param target_column: The name of the target column to predict.
        :param n_estimators: Number of trees in the forest (default is 100).
        :param model_name: Optional name for the model, used for saving.
        """
        self.target_column = target_column

        # Drop unnecessary columns, ensuring they exist
        features = data.drop(columns=[target_column, '_id', 'Type'], errors='ignore')  # Drop 'Type' too
        self.target = data[target_column]

        # One-hot encode categorical features
        self.features = pd.get_dummies(features, drop_first=True)
        logging.info(f"Features shape after encoding: {self.features.shape}")

        # Initialize the RandomForestClassifier model with n_estimators
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        logging.info(f"Initialized {self.model.__class__.__name__} with {n_estimators} estimators.")

        # Train the model
        self.train()

        # Store the instance in the models list
        Machine.models.append(self)

        # Save the model if a model name is provided
        if model_name:
            self.save(model_name)

    def train(self):
        """
        Trains the RandomForestClassifier model on the provided feature and target data.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        self.model.fit(X_train, y_train)

        # Evaluate model accuracy on test data
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model trained with accuracy: {accuracy:.2%}")

    def save(self, model_name: str):
        """
        Saves the trained model to the specified filepath.

        :param model_name: Name for the saved model (used for filename).
        """
        filepath = os.path.join(self.model_directory, f"{model_name}.joblib")
        logging.info(f"Saving model to: {filepath}")
        joblib.dump(self.model, filepath)
        logging.info(f"Model saved to {filepath}")

        # Verify if the model has been saved successfully
        if not os.path.exists(filepath):
            raise Exception(f"Failed to save the model at {filepath}")

    @classmethod
    def get_model_names(cls) -> List[str]:
        """
        Returns the names of the stored models based on .joblib files.

        :return: List of model names.
        """
        if not os.path.exists(cls.model_directory):
            logging.warning(f"Model directory {cls.model_directory} does not exist.")
            return []

        return [f[:-7] for f in os.listdir(cls.model_directory) if f.endswith('.joblib')]

    @classmethod
    def open(cls, model_name: str):
        """
        Loads a model from the specified model name, ensuring it is a RandomForestClassifier,
        and initializes a Machine instance.

        :param model_name: Name of the model to be loaded (without extension).
        :return: A Machine instance with the loaded model.
        """
        filepath = os.path.join(cls.model_directory, f"{model_name}.joblib")
        logging.info(f"Loading model from: {filepath}")

        try:
            model = joblib.load(filepath)
        except FileNotFoundError:
            logging.error(f"Model file {filepath} not found.")
            raise Exception(f"Model file {filepath} not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")
            raise Exception(f"An error occurred while loading the model: {e}")

        # Verify the model type is RandomForestClassifier
        if not isinstance(model, RandomForestClassifier):
            raise TypeError("Loaded model is not a RandomForestClassifier. Please check the saved model file.")

        machine_instance = cls.__new__(cls)  # Create instance without calling __init__
        machine_instance.model = model
        Machine.models.append(machine_instance)  # Add to models list
        logging.info(f"Model {model_name} loaded successfully.")
        return machine_instance

    def predict(self, features: DataFrame) -> Tuple[str, float]:
        """
        Makes predictions on the provided feature data.

        :param features: DataFrame containing the feature data for prediction.
        :return: Tuple containing the predicted class and its probability.
        """

        # One-hot encode the incoming features to match training data format
        features_encoded = pd.get_dummies(features, drop_first=True)

        # Reindex the columns to match the training feature set
        features_encoded = features_encoded.reindex(columns=self.features.columns, fill_value=0)

        # Make predictions
        prediction = self.model.predict(features_encoded)[0]
        probability = self.model.predict_proba(features_encoded).max()
        logging.info(f"Prediction: {prediction}, Probability: {probability:.2%}")
        return prediction, probability

    def predict_proba(self, features: DataFrame) -> List[List[float]]:
        """
        Returns the probabilities of each class for the provided feature data.

        :param features: DataFrame containing the feature data for prediction.
        :return: List of probabilities for each class.
        """

        # One-hot encode the incoming features to match training data format
        features_encoded = pd.get_dummies(features, drop_first=True)

        # Reindex the columns to match the training feature set
        features_encoded = features_encoded.reindex(columns=self.features.columns, fill_value=0)

        # Log the shape and columns of the encoded features for debugging
        logging.info(f"Encoded features shape: {features_encoded.shape}")
        logging.info(f"Encoded features columns: {features_encoded.columns.tolist()}")

        # Get the probabilities of each class
        probabilities = self.model.predict_proba(features_encoded)
        logging.info(f"Probabilities: {probabilities.tolist()}")

        return probabilities.tolist()  # Return as a list of probabilities

    def __call__(self, features: DataFrame) -> Tuple[str, float]:
        """
        Calls the predict method to make predictions on the provided feature data.

        :param features: DataFrame containing the feature data for prediction.
        :return: Tuple containing the predicted class and its probability.
        """
        # Call the predict method and return the prediction and its probability
        prediction, probability = self.predict(features)
        logging.info(f"Called predict: {prediction}, Probability: {probability:.2%}")
        return prediction, probability

    def info(self) -> str:
        """
        Returns information about the RandomForestClassifier model, including its parameters.

        :return: A formatted string with model details.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_info = f"Base Model: {self.model.__class__.__name__}\n"
        model_info += f"Description: RandomForestClassifier Model with {len(self.model.estimators_)} trees.\n"
        model_info += f"Timestamp: {timestamp}\n"
        model_info += "Parameters:\n" + "\n".join([f"{key}: {value}" for key, value in self.model.get_params().items()])
        
        logging.info("Model Info:")
        logging.info(model_info)
        
        return model_info
