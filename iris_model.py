import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()

# Mapping of iris class indices to names
iris_names = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

# Create DataFrame from the iris dataset
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the dataset into training and test sets
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2)

# Train the SVM model
model = SVC()
model.fit(xtr, ytr)

# Function to take in a list of parameters and return the predicted class name
def iris_predict(iris_params):
    # Create a DataFrame with feature names for the input
    input_df = pd.DataFrame([iris_params], columns=iris.feature_names)
    result = model.predict(input_df)
    return iris_names[result[0]]

