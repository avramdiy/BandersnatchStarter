import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

class Machine:

    def __init__(self, df, model_type="logistic"):
        """
        Initialize the machine learning model.
        :param df: DataFrame containing features and target.
        :param model_type: The type of model to use ('logistic' or 'random_forest').
        """
        # Ensure the target column exists
        if 'Rarity' not in df.columns:
            raise ValueError("Target column 'target' not found in the DataFrame.")

        # Split the DataFrame into features (X) and target (y)
        self.features = df.drop(columns=['Rarity'])  # Replace 'target' with the actual target column name
        self.target = df['Rarity']  # Replace 'target' with the actual target column name

        # Initialize the model based on the selected type
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=200)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier()
        else:
            raise ValueError("Invalid model type. Choose 'logistic' or 'random_forest'.")

        # Fit the model with the provided data
        self.model.fit(self.features, self.target)

    def predict(self, feature_basis):
        """
        Make a prediction based on the provided feature data.
        :param feature_basis: DataFrame containing the feature data.
        :return: A tuple of (prediction, probability).
        """
        # Ensure feature_basis has the correct columns
        feature_basis = feature_basis.reindex(columns=self.features.columns, fill_value=0)

        # Debugging: Print the feature_basis and its columns
        print("Feature basis for prediction:")
        print(feature_basis)

        # Predict the class
        prediction = self.model.predict(feature_basis)

        # Predict the probabilities
        probability = self.model.predict_proba(feature_basis)  # Get probabilities for all classes

        # Debugging: Print the prediction and probabilities
        print("Prediction:", prediction)
        print("Probabilities:", probability)

        # Return prediction and the max probability for the predicted class
        return prediction[0], max(probability[0])  # Return first element for prediction and max probability

    def save(self, filepath):
        """
        Save the trained model to a file.
        :param filepath: The file path where the model will be saved.
        """
        joblib.dump(self.model, filepath)

    @staticmethod
    def open(filepath):
        """
        Load a model from a file.
        :param filepath: The file path where the model is stored.
        :return: The loaded model.
        """
        return joblib.load(filepath)

    def info(self):
        """
        Return information about the model.
        :return: A dictionary containing model type and parameters.
        """
        return {
            "model_type": type(self.model).__name__,
            "params": self.model.get_params(),  # Get the model parameters
            "initialized_at": datetime.now()
        }
