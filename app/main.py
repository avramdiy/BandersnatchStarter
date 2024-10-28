import os
import pandas as pd
from flask import Flask, jsonify, render_template, request, redirect, url_for
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
from MonsterLab import Monster
from base64 import b64decode
from Fortuna import random_int, random_float
from app.data import Database
from app.graph import create_chart
from app.machine import Machine  # Updated Machine class
import numpy as np  # Ensure to import NumPy
import joblib  # Import joblib for loading models
import logging  # Import logging for error tracking

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# Initialize global data variable
loaded_data = None
model_directory = 'C:\\Users\\Ev\\Desktop\\Bandersnatch\\models'  # Update this path to point to a specific model directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    global loaded_data
    db = Database()
    loaded_data = db.dataframe()  # Load your DataFrame here

def load_models():
    """
    Load models from the model directory.
    """
    if not os.path.exists(model_directory):
        logging.error(f"Model directory '{model_directory}' does not exist.")
        return []

    models = []
    for model_file in os.listdir(model_directory):
        if model_file.endswith('.joblib'):
            model_path = os.path.join(model_directory, model_file)
            try:
                model = joblib.load(model_path)
                models.append(model)
                logging.info(f"Loaded model from: {model_path}")
            except Exception as e:
                logging.error(f"Error loading model from {model_path}: {e}")

    return models

@app.route("/")
def home():
    """
    Renders the home page.
    """
    load_data()  # Load data when accessing the home page
    return render_template(
        "home.html",
        sprint="Sprint 1",
        monster=Monster().to_dict(),
        password=b64decode(b"VGFuZ2luZ3lhdGlvbiBEcmVhbS=="),  # Corrected base64 encoding
    )

@app.route("/data")
def data_view():
    """
    Renders the data page which shows the count of documents and an HTML table of seed data.
    """
    db = Database()
    return render_template(
        "data.html",
        count=db.count(),
        table=db.html_table(),  # HTML table for seeds
    )

@app.route("/dataframe")
def dataframe_view():
    """
    Renders the page to display a DataFrame as an HTML table.
    """
    load_data()  # Ensure data is loaded
    table_html = loaded_data.to_html(classes='dataframe', header=True, index=False)  # Convert DataFrame to HTML
    return render_template("dataframe.html", table_html=table_html)

@app.route("/view", methods=["GET", "POST"])
def view():
    """
    Renders the view page with a graph based on selected axes and target.
    """
    load_data()  # Ensure data is loaded
    logging.info("Database connected")

    options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
    x_axis = request.values.get("x_axis") or options[1]
    y_axis = request.values.get("y_axis") or options[2]
    target = request.values.get("target") or options[4]

    # Generate the chart based on the selected axes and target
    graph = create_chart(
        df=loaded_data,
        x=x_axis,
        y=y_axis,
        target=target,
    ).to_json()

    return render_template(
        "view.html",
        options=options,
        x_axis=x_axis,
        y_axis=y_axis,
        target=target,
        count=Database().count(),
        graph=graph,
    )

@app.route('/model', methods=['GET', 'POST'])
def model():
    global loaded_data
    selected_model_info = None
    selected_model_probs = None  # To store probabilities
    prediction_result = None  # To store prediction result

    # Load models from the directory to populate the dropdown
    models = load_models()

    if request.method == 'POST':
        # Logic to create a new model
        try:
            num_trees = int(request.form['num_trees'])
            # Create a new instance of Machine with the specified number of trees
            if loaded_data is not None:
                model_name = f"model_{len(models) + 1}"  # Generate a model name
                machine_instance = Machine(loaded_data, target_column='Rarity', n_estimators=num_trees, model_name=model_name)
                models.append(machine_instance)  # Add new model to the list
                machine_instance.save(model_name)  # Save the new model

        except ValueError:
            logging.error("Invalid number of trees specified.")
            return redirect(url_for('model'))

        return redirect(url_for('model'))  # Redirect back to the model page to see updated models

    # Load models only if a specific model is selected from the dropdown
    if request.args.get('model_name'):
        selected_model_index = int(request.args.get('model_name').split(' ')[-1]) - 1  # Get index from the dropdown
        if 0 <= selected_model_index < len(models):
            selected_model = models[selected_model_index]

            # Get model information directly from the attributes
            selected_model_info = {
                "n_estimators": getattr(selected_model, 'n_estimators', 'N/A'),
                "max_depth": getattr(selected_model, 'max_depth', 'N/A'),
                "feature_importances": getattr(selected_model, 'feature_importances_', 'N/A'),
                "classes": getattr(selected_model, 'classes_', 'N/A'),
            }

            # Get the class probabilities for the selected model
            if loaded_data is not None:
                # Check the actual columns in loaded_data
                logging.info(f"Loaded data columns: {loaded_data.columns.tolist()}")

                # Drop the 'Name' column if it exists
                if 'Name' in loaded_data.columns:
                    loaded_data.drop(columns=['Name'], inplace=True)
                    logging.info("Dropped 'Name' column from loaded data.")

                # Generate feature names based on remaining columns
                feature_names = loaded_data.columns.tolist()  # Use remaining columns as feature names
                logging.info(f"Generated feature names: {feature_names}")

                # Filter loaded_data to only include relevant features
                try:
                    loaded_data = loaded_data[feature_names]  # Ensure we're filtering only relevant features
                except KeyError as e:
                    logging.error(f"KeyError: {str(e)}. Ensure that feature names match loaded_data columns.")
                    return render_template(
                        'model.html',
                        model_names=[f'Model {i + 1}' for i in range(len(models))],
                        selected_model_info=selected_model_info,
                        selected_model_probs=None,
                        prediction_result=None
                    )

                # Proceed with prediction
                try:
                    selected_model_probs = selected_model.predict_proba(loaded_data)  # Pass in the loaded data for probabilities

                    # Ensure the output is a NumPy array if it's a list
                    if isinstance(selected_model_probs, list):
                        selected_model_probs = np.array(selected_model_probs)

                    # Check if selected_model_probs is not None and has more than 0 elements
                    if selected_model_probs is not None and selected_model_probs.size > 0:
                        prediction_result = {
                            "class": selected_model.predict(loaded_data.iloc[[0]])[0],  # Get predicted class
                            "probability": selected_model_probs[0].max()  # Get maximum probability from the first sample
                        }
                    else:
                        prediction_result = {
                            "class": None,
                            "probability": None
                        }
                except Exception as e:
                    logging.error(f"Prediction error: {e}")
                    prediction_result = {
                        "class": None,
                        "probability": None
                    }

    return render_template(
        'model.html',
        model_names=[f'Model {i + 1}' for i in range(len(models))],  # Create model names dynamically
        selected_model_info=selected_model_info,
        selected_model_probs=selected_model_probs,
        prediction_result=prediction_result
    )



@app.route("/seed", methods=["POST"])
def seed():
    """
    Seeds the database with a specified number of Monster documents.
    """
    amount = request.form.get("amount", type=int)
    if amount:
        db = Database()
        db.seed(amount)
    return redirect(url_for('data_view'))  # Redirect to data_view

@app.route("/reset", methods=["POST"])
def reset():
    """
    Resets the database by deleting all documents.
    """
    db = Database()
    db.reset()
    return redirect(url_for('data_view'))  # Redirect to data_view

@app.route("/api/chart", methods=["GET"])
def chart_route():
    """
    API route to generate and return a chart as JSON.
    """
    x_axis = request.args.get('x')
    y_axis = request.args.get('y')
    target = request.args.get('target')

    db = Database()
    data = db.dataframe()  # Fetch the DataFrame from your database

    chart = create_chart(data, x_axis, y_axis, target)
    return jsonify(chart.to_json()), 200  # Return the serialized chart as JSON

if __name__ == '__main__':
    app.run()
