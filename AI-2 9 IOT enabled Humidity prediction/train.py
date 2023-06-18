import pandas as pd
from sklearn.linear_model import LinearRegression
import json

# Load the data from the CSV file
data = pd.read_csv('data.txt')

# Split the data into input (X) and output (y) variables
X = data[['Temperature']]
y = data['Humidity']

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# Extract the coefficients and intercept from the trained model
coefficients = model.coef_[0]
intercept = model.intercept_

# Create a dictionary to store the model data
model_data = {
    'coefficients': coefficients,
    'intercept': intercept
}

# Save the model data to a JSON file
with open('model.json', 'w') as f:
    json.dump(model_data, f)
