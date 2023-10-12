# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

def eda(df, target_names, feature_names, target):
    st.subheader("Exploratry Data Analasis and Visualization")
    st.write("Choose a plot type from the option below")
    # Add option to show/hide data details
    if st.checkbox("Show raw data"):
        st.write(df)
    # Add option to show/hide missing values
    if st.checkbox("Show missing values:"):
        st.write(df.isna().sum())
    if st.checkbox("Show Data types"):
        st.write(df.dtypes)
    if st.checkbox("Show descriptive Stastistics"):
        st.write(df.describe())
    if st.checkbox("Show correlation Matrix"):
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm")
        st.pyplot()
    if st.checkbox("Show histogram for each attributes"):
        for col in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=20, density=True, alpha=0.5)
            ax.set_title(col)
            st.pyplot(fig)
    if st.checkbox("Show Density for each attributes"):
        for col in df.columns:
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], fill=True)
            ax.set_title(col)
            st.pyplot(fig)
    if st.checkbox("Show Scatter plot"):
        fig = px.scatter(df, x = feature_names[0], y = feature_names[1], color = target)
        st.plotly_chart(fig)