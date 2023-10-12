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
import pickle
import plotly.express as px

def ml(df, target):
    st.subheader("Predictive Modelling")
    st.write("Choose a transform type and Model from the option below")
    
    X = df.values
    Y = target.values
    test_proportion = 0.30
    seed = 5

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_proportion, random_state=seed)
    transform_options = ["None",
                        "StandardScaler",
                        "Normalizer",
                        "MinMaxScaler"]
    
    transform = st.selectbox("Select data transform",
                             transform_options)
    if transform == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #save transform
        standardScaler_file = "standardScaler.pickle"
        pickle.dump(scaler, open(standardScaler_file, "wb"))

    elif transform == "Normalizer":
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #save transform
        normalizerScaler_file = "normalizerScaler.pickle"
        pickle.dump(scaler, open(normalizerScaler_file, "wb"))
        
    elif transform == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #save transform
        minMaxScaler_file = "minMaxScaler.pickle"
        pickle.dump(scaler, open(minMaxScaler_file, "wb"))
        
    else:
        X_train = X_train
        X_test = X_test
    
    classifier_list = ["LogisticRegression",
                       "SVM", 
                       "DecisionTree", 
                       "KNeighbors",
                       "RandomForest"]
    classifier = st.selectbox("Select Classifier", classifier_list)

    # Add option to select classifier
    #add LogisticRegression
    if classifier == "LogisticRegression":
        st.write("Here are the result of a Logistic Regression")
        solver_value = st.selectbox("Select Solver", 
                                      ["lbfgs",
                                      "liblinear",
                                      "newton-cg",
                                      "newton-cholesky"])
        model = LogisticRegression(solver=solver_value)
        model.fit(X_train, Y_train)

        #Make prediction
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='micro')
        recall = recall_score(Y_test, Y_pred, average='macro')
        f1 = f1_score(Y_test, Y_pred, average='weighted')

        # display details 
        st.write(f'Accuracy : {accuracy}')
        st.write(f'Precision : {precision}')
        st.write(f'recall : {recall}')
        st.write(f'f1 : {f1}')
        st.write("confusion Matrix")
        st.write(confusion_matrix(Y_test, Y_pred))

        #save model
        logisticRegressionModel_file = "LogisticRegressionModel.pickle"
        pickle.dump(model, open(logisticRegressionModel_file, "wb"))

    elif classifier == "DecisionTree":
        st.write("Here are the result of a Decision Tree")
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        #Make prediction
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='micro')
        recall = recall_score(Y_test, Y_pred, average='macro')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        # display details 
        st.write(f'Accuracy : {accuracy}')
        st.write(f'Precision : {precision}')
        st.write(f'recall : {recall}')
        st.write(f'f1 : {f1}')
        st.write("confusion Matrix")
        st.write(confusion_matrix(Y_test, Y_pred))
        #save model
        decisionTreeModel_file = "DecisionTreeModel.pickle"
        pickle.dump(model, open(decisionTreeModel_file, "wb"))
