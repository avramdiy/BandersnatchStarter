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
    Load models from the model directory, returning both models and their names.
    """
    if not os.path.exists(model_directory):
        logging.error(f"Model directory '{model_directory}' does not exist.")
        return [], []

    models = []
    model_names = []  # List to store the model names

    for model_file in os.listdir(model_directory):
        if model_file.endswith('.joblib'):
            model_path = os.path.join(model_directory, model_file)
            try:
                model = joblib.load(model_path)
                model_name = model_file.split('.')[0]  # Use the file name as the model name
                models.append(model)
                model_names.append(model_name)  # Store the model name
                logging.info(f"Loaded model '{model_name}' from: {model_path}")
            except Exception as e:
                logging.error(f"Error loading model from {model_path}: {e}")

    return models, model_names  # Return both the models and their names


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
    selected_model_probs = None
    prediction_result = None

    # Load available models with custom names
    models, model_names = load_models()  # Now loading both models and names

    if request.method == 'POST':
        # Handle model creation with specified trees
        try:
            num_trees = int(request.form['num_trees'])
            model_name = request.form['model_name'].strip()  # Get custom model name from form
            if not model_name:
                model_name = f"model_{len(models) + 1}.joblib"  # Default name if no custom name is provided

            if loaded_data is not None:
                machine_instance = Machine(loaded_data, target_column='Rarity', n_estimators=num_trees, model_name=model_name)
                models.append(machine_instance)
                machine_instance.save(model_name)  # Save the model with the custom name
                logging.info(f"Model '{model_name}' created successfully.")
        except ValueError:
            logging.error("Invalid number of trees specified.")
            return redirect(url_for('model'))

        return redirect(url_for('model'))  # Reload page to show the updated list of models

    # Process model selection from dropdown
    selected_model_name = request.args.get('model_name')
    selected_model_index = None
    if selected_model_name:
        # Find the selected model index based on the custom name
        selected_model_index = model_names.index(selected_model_name)
        if 0 <= selected_model_index < len(models):
            selected_model = models[selected_model_index]

            # Capture basic model information
            selected_model_info = {
                "n_estimators": getattr(selected_model, 'n_estimators', 'N/A'),
                "max_depth": getattr(selected_model, 'max_depth', 'N/A'),
                "feature_importances": getattr(selected_model, 'feature_importances_', 'N/A'),
                "classes": getattr(selected_model, 'classes_', 'N/A'),
            }

            # Ensure the data only contains feature columns
            feature_columns = loaded_data.drop(columns=['Rarity', '_id', 'Type', 'Name'], errors='ignore')
            try:
                selected_model_probs = selected_model.predict_proba(feature_columns)
                prediction_result = {
                    "class": selected_model.predict(feature_columns.iloc[[0]])[0],
                    "probability": selected_model_probs[0].max() if selected_model_probs.size > 0 else None
                }
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                prediction_result = {"class": None, "probability": None}

    # Render the template with models and prediction information
    return render_template(
        'model.html',
        model_names=model_names,  # Pass the custom names here
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
