# Iris Flower Species Predictor 

This project is a machine learning application that predicts the species of an Iris flower, **Setosa**, **Versicolor**, or **Virginica** based on user inputted measurements of **sepal length**, **sepal width**, **petal length**, and **petal width**.

It is built using **Python**, **Pandas**, **Scikit-learn**, **Seaborn**, **Matplotlib**, and **Streamlit** for interactive UI. The model is trained on the classic **Iris dataset**, and uses **Logistic Regression** for classification.

## 🔍 Features,

- 📊 Visualizes feature correlations using a heatmap.
- 📈 Displays a confusion matrix to assess model accuracy.
- 🔁 Converts species names to numeric labels via encoding.
- 🧠 Uses logistic regression to classify flower species.
- 🎯 Offers real-time prediction based on user inputs.
- 🌼 Displays species-specific images after prediction.

## 📦 Dataset

- Dataset: [Iris Dataset from UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- Format: CSV (zipped in `archive.zip`)
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
  - Species (target)

## ⚙️ How It Works

1. The dataset is loaded and cleaned (removes ID column).
2. Correlation matrix is generated for feature insights.
3. Dataset is split into training and testing sets.
4. Labels are encoded using `LabelEncoder`.
5. Logistic Regression model is trained on training data.
6. Model predicts species based on new measurements.
7. A relevant flower image is displayed for the predicted class.

## 📸 Species Images

- **Setosa**: ![Setosa](https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg)
- **Versicolor**: ![Versicolor](https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg)
- **Virginica**: ![Virginica](https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg)

## ▶️ Running the Project

Make sure you have Python 3.7+ installed.

### 1. Install Dependencies

```bash
pip install -r requirements.txt

