import altair as alt
import pandas as pd
from pandas import DataFrame
from bson import ObjectId


def create_chart(df: DataFrame, x: str, y: str, target: str) -> alt.Chart:
    """
    Generates an Altair chart based on the provided DataFrame and column names.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame containing the data to be plotted.
    x : str
        The name of the column to be plotted on the x-axis.
    y : str
        The name of the column to be plotted on the y-axis.
    target : str
        The name of the column used for color encoding and tooltip information.

    Returns
    -------
    alt.Chart
        The generated Altair Chart object.
    """

    # Convert ObjectId to string in the specified columns if necessary
    for col in [target, x, y]:
        if col in df.columns and df[col].apply(lambda x: isinstance(x, ObjectId)).any():
            df[col] = df[col].astype(str)

    # Create the chart object using Altair, with circular marks for data points
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(x, title=x),  # Encode the x-axis with the provided column
        y=alt.Y(y, title=y),  # Encode the y-axis with the provided column
        color=alt.Color(target, legend=alt.Legend(title=target)),  # Color points by the target column
        tooltip=[x, y, target]  # Show tooltips with x, y, and target columns
    ).properties(
        title=f"Plot of {y} vs {x}",  # Set the title of the chart
        width=500,  # Define the width of the chart
        height=300,  # Define the height of the chart
        background='transparent',  # Set the background as transparent
        padding=5  # Define the padding around the chart
    ).configure(
        axis={"labelColor": "white", "titleColor": "white"},  # Set axis labels and titles to white
        title={"color": "white"}  # Set the title color to white
    ).interactive()  # Enable interactive features

    return chart  # Return the created Altair chart object
