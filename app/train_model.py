import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from app.data import Database

def train_model():
    """
    Train machine learning models and report their accuracy.
    """
    db = Database()
    
    # Fetch the DataFrame from the database
    df = db.dataframe()
    
    # Print features in the DataFrame
    if not df.empty:
        features = df.columns.tolist()  # Get the feature names
        print("Features in the database:", features)
    else:
        print("The database is empty.")
        return

    # Assuming you want to use specific features for training
    feature_columns = ['Level', 'Health', 'Energy', 'Sanity']  # Update this list based on your needs
    target_column = 'Rarity'

    # Ensure that the specified features exist in the DataFrame
    if not all(col in features for col in feature_columns):
        print("One or more feature columns are missing from the DataFrame.")
        return

    # Prepare the feature set and target variable
    X = df[feature_columns]
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=500)  # Increased max_iter
    }

    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        # Train the model
        if model_name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        
        # Check if it's the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name

    print(f"The best model is {best_model} with an accuracy of {best_accuracy:.4f}")

if __name__ == "__main__":
    train_model()
