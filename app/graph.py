import altair as alt
from pandas import DataFrame

def chart(df: DataFrame, x: str, y: str, target: str) -> alt.Chart:
    """
    Generates an Altair chart based on the provided DataFrame and column names.

    Parameters:
    ----------
    df : DataFrame
        The pandas DataFrame containing the data to be plotted.
    x : str
        The name of the column to be plotted on the x-axis.
    y : str
        The name of the column to be plotted on the y-axis.
    target : str
        The name of the column used for color encoding and tooltip information.

    Returns:
    -------
    alt.Chart
        The generated Altair Chart object.
    """
    
    # Create the chart object using Altair, with circular marks for data points
    chart = alt.Chart(df).mark_circle().encode(
        x=x,  # Encode the x-axis with the provided column
        y=y,  # Encode the y-axis with the provided column
        color=alt.Color(target, legend=alt.Legend(title="Target")),  # Color points by the target column
        tooltip=[x, y, target]  # Show tooltips with x, y, and target columns
    ).properties(
        title=f"Plot of {y} vs {x}",  # Set the title of the chart
        width=500,  # Define the width of the chart
        height=300,  # Define the height of the chart
        background='transparent',  # Set the background as transparent to match the dark theme
        padding=5  # Define the padding around the chart
    ).configure(
        axis={"labelColor": "white", "titleColor": "white"},  # Set axis labels and titles to white for dark theme
        title={"color": "white"}  # Set the title color to white to match the theme
    )

    return chart  # Return the created Altair chart object
