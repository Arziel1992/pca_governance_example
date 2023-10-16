"""
This module contains a Streamlit app that performs principal component analysis (PCA) on a set of skills data.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd


def calculate_vector_magnitude(time: float, cost: float, complexity: float) -> float:
    """
    Calculates the magnitude of a vector with given time, cost and complexity.

    Args:
    time (float): Time value of the vector.
    cost (float): Cost value of the vector.
    complexity (float): Complexity value of the vector.

    Returns:
    float: Magnitude of the vector.
    """
    return np.sqrt(time**2 + cost**2 + complexity**2)


def calculate_point_magnitude(coords: np.ndarray) -> np.ndarray:
    """
    Calculates the magnitude of a point with given coordinates.

    Args:
    coords (np.ndarray): Array of coordinates of the point.

    Returns:
    np.ndarray: Magnitude of the point.
    """
    return np.sqrt(np.sum(np.square(coords), axis=1))


skills_data = {
    'Programming (Python)': (6, 3, 5),
    'Data Governance': (8, 7, 9),
    'Statistical Analysis': (7, 4, 8),
    'Machine Learning Algorithms': (9, 5, 9),
    'Healthcare Policy Understanding': (8, 6, 7),
    'Database Management': (5, 4, 6),
    'Cybersecurity': (7, 8, 8),
    'Project Management': (6, 5, 7),
    'Ethical Considerations in AI': (7, 6, 9),
    'Cloud Computing': (5, 7, 6)
}

magnitudes = {skill: calculate_vector_magnitude(
    *values) for skill, values in skills_data.items()}

colors = [f'rgba({r}, {g}, {b}, 1)' for r, g, b, _ in (
    np.random.rand(len(skills_data), 4) * 255).astype(int)]

# Create a 3D scatter plot for each skill with time, cost and complexity as axes
traces = [
    go.Scatter3d(
        x=[0, time],
        y=[0, cost],
        z=[0, complexity],
        mode='lines',
        name=skill,
        line=dict(color=color, width=6)
    )
    for (skill, (time, cost, complexity)), color in zip(skills_data.items(), colors)
]

# Create a figure with the 3D scatter plots
fig = go.Figure(data=traces)

# Update the layout of the figure
fig.update_layout(
    scene=dict(
        xaxis_title='Time',
        yaxis_title='Cost',
        zaxis_title='Complexity',
        aspectmode='cube',
    ),
    margin=dict(l=0, r=0, b=0, t=0),
)

# Display the figure in the Streamlit app
st.plotly_chart(fig)

# Create a table with the magnitudes, time, cost and complexity of each skill
table_data = {
    'Magnitude': list(magnitudes.values()),
    'Time': [values[0] for values in skills_data.values()],
    'Cost': [values[1] for values in skills_data.values()],
    'Complexity': [values[2] for values in skills_data.values()]
}
table_df = pd.DataFrame(table_data, index=list(skills_data.keys()))

# Display the table in the Streamlit app
st.write(table_df)

# Perform PCA on the skills data and display the results
data_matrix = np.array(list(skills_data.values()))

explained_variances = []
for n_components in range(2, 0, -1):
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data_matrix)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    explained_variances.append((n_components, explained_variance))

    columns = [f'PC{i+1}' for i in range(n_components)]
    transformed_df = pd.DataFrame(
        transformed_data, columns=columns, index=list(skills_data.keys()))

    # Create a scatter plot of the transformed data with the principal components as axes
    fig = px.scatter(transformed_df, x='PC1', y='PC2' if n_components > 1 else None,
                     labels={
                         f'PC{i+1}': f'Principal Component {i+1}' for i in range(n_components)},
                     title=f'PCA Result ({n_components}D)',
                     color=transformed_df.index,
                     color_discrete_sequence=colors)
    st.write(fig)

    # Create a table with the magnitudes and principal components of each skill
    magnitudes = calculate_point_magnitude(transformed_data)
    display_df = pd.DataFrame(
        {'Magnitude': magnitudes}, index=list(skills_data.keys()))
    for i, column in enumerate(columns):
        display_df[column] = transformed_data[:, i]
    st.write(display_df)

# Create a table with the explained variances of each principal component
pca_df = pd.DataFrame(explained_variances, columns=[
                      'Dimensions', 'Explained Variance'])

# Display the table in the Streamlit app
st.write("### Principal Component Analysis (PCA) Results:")
st.write(pca_df)
