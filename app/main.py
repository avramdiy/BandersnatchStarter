import os
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
from MonsterLab import Monster
from pandas import DataFrame
from base64 import b64decode
from Fortuna import random_int, random_float
from app.data import Database
from app.graph import chart
from app.machine import Machine

# Load environment variables from the .env file
load_dotenv()

SPRINT = 1
app = Flask(__name__)

@app.route("/")
def home():
    """
    Renders the home page.

    :return: The rendered HTML template for the home page with sprint number, 
             a Monster object converted to a dictionary, and a decoded password.
    """
    return render_template(
        "home.html",
        sprint=f"Sprint {SPRINT}",
        monster=Monster().to_dict(),
        password=b64decode(b"VGFuZ2VyaW5lIERyZWFt"),
    )

@app.route("/data")
def data():
    """
    Renders the data page which shows the count of documents and an HTML table of seed data.

    :return: The rendered HTML template for the data page with document count 
             and HTML table of seeds if SPRINT >= 1, otherwise just the template.
    """
    if SPRINT < 1:
        return render_template("data.html")
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

    :return: The rendered HTML template for the DataFrame view with the DataFrame 
             converted to an HTML table.
    """
    db = Database()
    df = db.dataframe()  # Get the DataFrame
    table_html = df.to_html(classes='dataframe', header=True, index=False)  # Convert DataFrame to HTML
    return render_template("dataframe.html", table_html=table_html)

@app.route("/view", methods=["GET", "POST"])
def view():
    """
    Renders the view page with a graph based on selected axes and target.

    :return: The rendered HTML template for the view page with graph data and options,
             if SPRINT >= 2, otherwise just the template.
    """
    if SPRINT < 2:
        return render_template("view.html")
    db = Database()
    options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
    x_axis = request.values.get("x_axis") or options[1]
    y_axis = request.values.get("y_axis") or options[2]
    target = request.values.get("target") or options[4]
    graph = chart(
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

    :return: The rendered HTML template for the model page with prediction results, 
             confidence, and model information if SPRINT >= 3, otherwise just the template.
    """
    if SPRINT < 3:
        return render_template("model.html")
    db = Database()
    options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
    filepath = os.path.join("app", "model.joblib")
    if not os.path.exists(filepath):
        df = db.dataframe()
        machine = Machine(df[options])
        machine.save(filepath)
    else:
        machine = Machine.open(filepath)
    stats = [round(random_float(1, 250), 2) for _ in range(3)]
    level = request.values.get("level", type=int) or random_int(1, 20)
    health = request.values.get("health", type=float) or stats.pop()
    energy = request.values.get("energy", type=float) or stats.pop()
    sanity = request.values.get("sanity", type=float) or stats.pop()
    prediction, confidence = machine(DataFrame(
        [dict(zip(options, (level, health, energy, sanity)))])
    )
    info = machine.info()
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

    :return: Redirects to the data page if SPRINT >= 1, otherwise redirects to the home page.
    """
    if SPRINT < 1:
        return redirect(url_for('home'))  # Redirect if SPRINT < 1
    amount = request.form.get("amount", type=int)
    if amount:
        db = Database()
        db.seed(amount)
    return redirect(url_for('data'))

@app.route("/reset", methods=["POST"])
def reset():
    """
    Resets the database by deleting all documents.

    :return: Redirects to the data page if SPRINT >= 1, otherwise redirects to the home page.
    """
    if SPRINT < 1:
        return redirect(url_for('home'))  # Redirect if SPRINT < 1
    db = Database()
    db.reset()
    return redirect(url_for('data'))

if __name__ == '__main__':
    app.run()
