import pickle
import numpy as np
import streamlit as st
import ml

standardScaler = pickle.load(open("standardScaler.pickle", "rb"))
normalScaler = pickle.load(open("normalizerScaler.pickle", "rb"))
minMaxScaler = pickle.load(open("minMaxScaler.pickle", "rb"))
logisticRegression = pickle.load(open('LogisticRegressionModel', 'rb'))
decisionTree = pickle.load(open("DecisionTreeModel.pickle"), "rb")
svm = pickle.load(open("SVMModel.pickle"), "rb")
kNeighbors = pickle.load(open("kNeighborsModel.pickle"), "rb")
randomForest = pickle.load(open("DecisionTreeModel.pickle"), "rb")

st.subheader("Predictive Modelling")
st.write("Choose a transform type and Model from the option below")
transform_options = ["None",
                        "StandardScaler",
                        "Normalizer",
                        "MinMaxScaler"]
classifier_list = ["LogisticRegression",
                       "SVM", 
                       "DecisionTree", 
                       "KNeighbors",
                       "RandomForest"]

transform = st.selectbox("Select data transform",
                             transform_options)
if transform == "StandardScaler":

    classifier = st.selectbox("Select Classifier", classifier_list)
    if classifier == "LogisticRegression":
        new_predict_1 = logisticRegression.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "DecisionTree":
        new_predict_1 = decisionTree.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "SVM":
        new_predict_1 = svm.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "KNeighbors":
        new_predict_1 = kNeighbors.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "RandomForest":
        new_predict_1 = randomForest.predict(standardScaler.transform())
        st.write(new_predict_1)
elif transform == "Normalizer":

    classifier = st.selectbox("Select Classifier", classifier_list)
    if classifier == "LogisticRegression":
        new_predict_1 = logisticRegression.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "DecisionTree":
        new_predict_1 = decisionTree.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "SVM":
        new_predict_1 = svm.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "KNeighbors":
        new_predict_1 = kNeighbors.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "RandomForest":
        new_predict_1 = randomForest.predict(standardScaler.transform())
        st.write(new_predict_1)

elif transform == "MinMaxScaler":

    classifier = st.selectbox("Select Classifier", classifier_list)
    if classifier == "LogisticRegression":
        new_predict_1 = logisticRegression.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "DecisionTree":
        new_predict_1 = decisionTree.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "SVM":
        new_predict_1 = svm.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "KNeighbors":
        new_predict_1 = kNeighbors.predict(standardScaler.transform())
        st.write(new_predict_1)
    elif classifier == "RandomForest":
        new_predict_1 = randomForest.predict(standardScaler.transform())
        st.write(new_predict_1)


 #       st.write(f"Accuracy: {accuracy}")
  #      st.write(f"Precision: {precision}")
   #     st.write(f"Recall: {recall}")
    #    st.write(f"F1 score: {f1}")
     #   st.write("Confusion Matrix:")
      #  st.write(confusion_matrix(y_test, y_pred))