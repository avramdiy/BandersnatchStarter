import os
import pandas as pd
from flask import Flask, jsonify, render_template, request, redirect, url_for
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
from MonsterLab import Monster
from pandas import DataFrame
from base64 import b64decode
from Fortuna import random_int, random_float
from app.data import Database
from app.graph import create_chart
from app.machine import Machine  # Updated Machine class

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    """
    Renders the home page.
    """
    return render_template(
        "home.html",
        sprint="Sprint 1",
        monster=Monster().to_dict(),
        password=b64decode(b"VGFuZ2VyaW5lIERyZWFt"),
    )

@app.route("/data")
def data():
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
    db = Database()
    df = db.dataframe()  # Get the DataFrame
    table_html = df.to_html(classes='dataframe', header=True, index=False)  # Convert DataFrame to HTML
    return render_template("dataframe.html", table_html=table_html)

@app.route("/view", methods=["GET", "POST"])
def view():
    """
    Renders the view page with a graph based on selected axes and target.
    """
    db = Database()
    print("Database connected")

    options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
    x_axis = request.values.get("x_axis") or options[1]
    y_axis = request.values.get("y_axis") or options[2]
    target = request.values.get("target") or options[4]
    
    # Generate the chart based on the selected axes and target
    graph = create_chart(
        df=db.dataframe(),
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
        count=db.count(),
        graph=graph,
    )

@app.route("/model", methods=["GET", "POST"])
def model():
    """
    Renders the model page with a prediction based on user input or random values.
    """
    db = Database()
    options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
    filepath = os.path.join("app", "model.joblib")
    
    # Check if model file exists
    if not os.path.exists(filepath):
        df = db.dataframe()  # Assuming this returns a DataFrame with a 'target' column
        # Train the model using the updated Machine class
        machine = Machine(df[options])  # Ensure df contains the correct columns
        machine.save(filepath)  # Save the trained model to model.joblib
    else:
        machine = Machine.open(filepath)  # Load the trained model from file

    # Generate random or user-supplied stats for prediction
    stats = [round(random_float(1, 250), 2) for _ in range(3)]
    level = request.values.get("level", type=int) or random_int(1, 20)
    health = request.values.get("health", type=float) or stats.pop()
    energy = request.values.get("energy", type=float) or stats.pop()
    sanity = request.values.get("sanity", type=float) or stats.pop()
    
    # Create a DataFrame for the feature basis
    feature_basis = DataFrame([{
        'Level': level,
        'Health': health,
        'Energy': energy,
        'Sanity': sanity
    }])

    # Debugging: Print feature_basis
    print("Feature basis DataFrame:")
    print(feature_basis)

    # Use the Machine class to make predictions
    try:
        prediction, confidence = machine.predict(feature_basis)  # Call the predict method
    except Exception as e:
        print("Error during prediction:", e)
        return "Error during prediction", 500

    info = machine.info()  # Get information about the model
    
    return render_template(
        "model.html",
        info=info,
        level=level,
        health=health,
        energy=energy,
        sanity=sanity,
        prediction=prediction,
        confidence=f"{confidence:.2%}",
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
    return redirect(url_for('data'))

@app.route("/reset", methods=["POST"])
def reset():
    """
    Resets the database by deleting all documents.
    """
    db = Database()
    db.reset()
    return redirect(url_for('data'))

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
