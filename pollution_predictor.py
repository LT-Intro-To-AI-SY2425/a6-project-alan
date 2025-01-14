import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("pollution.csv")

# Extract features and target variable
x = data[["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]].values
y = data["AQI Value"].values

# Split the data into training and testing sets (80% training, 20% testing)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Standardize the features (important for multivariable regression)
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Create the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(xtrain, ytrain)

# Get model coefficients and intercept
coef = np.around(model.coef_, 2)
intercept = round(model.intercept_, 2)
r_squared = round(model.score(xtest, ytest), 2)

print(f"The linear equation is: y = {coef[0]} * CO + {coef[1]} * Ozone + {coef[2]} * NO2 + {coef[3]} * PM2.5 + {intercept}")
print(f"R^2 value is: {r_squared}")

# Make predictions on the test set
predict = model.predict(xtest)
predict = np.around(predict, 2)

# Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(ytest, predict)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot actual vs predicted values
plt.scatter(ytest, predict)
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], color='red', linestyle='--')
plt.xlabel('Actual AQI Values')
plt.ylabel('Predicted AQI Values')
plt.title('Actual vs Predicted AQI Values')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("pollution.csv")

# Extract features and target variable
x = data[["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]].values
y = data["AQI Value"].values

# Split the data into training and testing sets (80% training, 20% testing)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Standardize the features (important for multivariable regression)
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Create the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(xtrain, ytrain)

# Get model coefficients and intercept
coef = np.around(model.coef_, 2)
intercept = round(model.intercept_, 2)
r_squared = round(model.score(xtest, ytest), 2)

print(f"The linear equation is: y = {coef[0]} * CO + {coef[1]} * Ozone + {coef[2]} * NO2 + {coef[3]} * PM2.5 + {intercept}")
print(f"R^2 value is: {r_squared}")

# Make predictions on the test set
predict = model.predict(xtest)
predict = np.around(predict, 2)

# Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(ytest, predict)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot actual vs predicted values
plt.scatter(ytest, predict)
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], color='red', linestyle='--')
plt.xlabel('Actual AQI Values')
plt.ylabel('Predicted AQI Values')
plt.title('Actual vs Predicted AQI Values')
plt.show()

