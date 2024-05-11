import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import urllib.request

'''Creating sample data for the Houseprice dataset
   Boston House dataset not available in scikit learn
   because i created a customed dataset'''
Houseprice_data = {
    "longitude": [-122.23, -122.22, -122.24, -122.25, -122.26, -122.23, -122.22, -122.24, -122.25, -122.26, 
                  -122.23, -122.22, -122.24, -122.25, -122.26, -122.23, -122.22, -122.24, -122.25, -122.26],
    "latitude": [37.88, 37.89, 37.90, 37.91, 37.92, 37.88, 37.89, 37.90, 37.91, 37.92, 
                 37.88, 37.89, 37.90, 37.91, 37.92, 37.88, 37.89, 37.90, 37.91, 37.92],
    "housing_median_age": [41, 21, 52, 36, 30, 41, 21, 52, 36, 30, 41, 21, 52, 36, 30, 41, 21, 52, 36, 30],
    "total_rooms": [880, 712, 1466, 1274, 1627, 880, 712, 1466, 1274, 1627, 880, 712, 1466, 1274, 1627, 
                    880, 712, 1466, 1274, 1627],
    "total_bedrooms": [129, 110, 190, 235, 280, 129, 110, 190, 235, 280, 129, 110, 190, 235, 280, 
                        129, 110, 190, 235, 280],
    "population": [322, 240, 496, 558, 565, 322, 240, 496, 558, 565, 322, 240, 496, 558, 565, 
                   322, 240, 496, 558, 565],
    "households": [126, 113, 226, 262, 295, 126, 113, 226, 262, 295, 126, 113, 226, 262, 295, 
                   126, 113, 226, 262, 295],
    "median_income": [8.3252, 8.3014, 7.2574, 5.6431, 3.8462, 8.3252, 8.3014, 7.2574, 5.6431, 3.8462, 
                      8.3252, 8.3014, 7.2574, 5.6431, 3.8462, 8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
    "median_house_value": [452600, 358500, 352100, 341300, 342200, 452600, 358500, 352100, 341300, 342200, 
                            452600, 358500, 352100, 341300, 342200, 452600, 358500, 352100, 341300, 342200]
}

# Converting the dictionary into DataFrame
Houseprice_df = pd.DataFrame(Houseprice_data)

# Save the DataFrame to CSV file
Houseprice_df.to_csv("Housepricepred.csv", index=False)

# Load the dataset
data = pd.read_csv("Housepricepred.csv")

# Spliting the dataset into X and y
X = data.drop(columns=["median_house_value"])
y = data["median_house_value"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate mean squared error (MSE) on testing set
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot predicted vs actual prices
plt.scatter(y_test, y_pred, color ='red', marker='*')
plt.plot(y_test,y_pred, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
