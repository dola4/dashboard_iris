# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
from ead import eda
from ml import ml
st.set_option('deprecation.showPyplotGlobalUse', False)

# load data
dataset = load_iris()

# create dataframe with iris data
print(dataset)
data = dataset.data
feature_names = dataset.feature_names # colunms (petal width ...), X
target_names = dataset.target_names # classes (setosa versicolor virginica)
target = dataset.target # output

df = pd.DataFrame(data, columns = feature_names)
# Make target a serie
target = pd.Series(target)
# streamlit 
# Set up app
st.set_page_config(page_title="EDA and ML Dashboard",
                   layout="centered",
                   initial_sidebar_state="auto")
#add Title and Markdown description
st.title("EDA and predictive Dashboard")
#define sidebar and sidebar option
option = ["EDA", "Predictive Modelling"]
selected_option = st.sidebar.selectbox("select an option", option)

# Do EDA
if selected_option == "EDA":
    eda(df, target_names, feature_names, target)

# Predictive Modelling
elif selected_option == "Predictive Modelling":
    ml(df, target)
