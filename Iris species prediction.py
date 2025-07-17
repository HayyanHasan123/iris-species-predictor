import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:\\Users\\T L S\\Downloads\\archive.zip")

print(df.info()) #basic info of file
print(df.head()) #prints first 5 rows
print(df.describe()) #gives basic info of the dataframe
print(df.isnull().sum()) #Tells us how many null values there are

df = df.drop('Id', axis = 1) #Had to drop the ID coulmn, cuz it does not have much use in ML models
print("The number of species are", df['Species'].value_counts())


#Using ONE HOT ENCODING to convert non numeric values(Like species name) into numeric values for ML model
one_hot_enc = pd.get_dummies(df, drop_first = True)
co_relation_matrix = one_hot_enc.corr()
# 1 & -1 means strong correlation
# near 0 means weak correlation
print(co_relation_matrix)

# Using Matplotlib for graphical visualization of correlation matrix
plt.figure(figsize = (12,8))
sns.heatmap(co_relation_matrix,
            annot = True,
            cmap = "coolwarm",
            fmt =".2f",
            cbar = True
)
plt.title("Iris Feature Collection Heat Map")
plt.show()

# Model Training and Evaluation
# In this section, we will train the Linear Regression model on the training dataset
# and evaluate its performance on both training and testing datasets.
# The data will be split into 2 parts, 80% for Training & 20% for Testing

x = df.drop("Species", axis = 1)
y = df["Species"]

# Now Splitting the Data
#Using any Number as Random State so the model's results remain consistent every time the code is run
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2) 

# Now converting the string based species column in y_test, y_train into numerical values using LabelEncoder
label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train)
y_test = label_enc.transform(y_test)

#Creating and training the prediction model
from sklearn.linear_model import LogisticRegression
pred_model = LogisticRegression(max_iter= 200)
pred_model.fit(x_train, y_train)

#Predicting The species
y_pred = pred_model.predict(x_test)

#Now determining the accuracy of the models predictions
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))

#Heat Map of the Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot = True,
            cmap = "Blues",
            fmt = "d"
)
plt.title("Confusion Matrix HeatMap for IRIS species Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# StreamLit App(UI for project)
st.set_page_config(page_title="Iris Flower Species Predictor", layout="centered")
st.title("Iris Flower Species Predictor")
st.write("Enter Flower Measurements Below")

sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction = pred_model.predict(input_data)
    predicted_species = label_enc.inverse_transform(prediction)

    st.success(f"Predicted Species: {predicted_species[0]}")

    if predicted_species[0] == "Iris-setosa":
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
             caption="Iris Setosa", use_container_width=True)

    elif predicted_species[0] == "Iris-versicolor":
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
             caption="Iris Versicolor", use_container_width=True)

    elif predicted_species[0] == "Iris-virginica":
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
             caption="Iris Virginica", use_container_width=True)