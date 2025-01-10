import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("pollution.csv")
x = data[["CO AQI Value","Ozone AQI Value","NO2 AQI Value","PM2.5 AQI Value"]].values
y = data["AQI Value"].values

#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size= 0.2)
#create linear regression model
model = LinearRegression().fit(xtrain, ytrain)
#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coef = np.around(model.coef_, 2)
intercept =round((model.intercept_), 2)
r_squared = round(model.score(x,y),2)

print(f"The linear equation is y={coef[0]}x + {coef[1]}x2 + {coef[2]}x3 + {coef[3]}x4 + {intercept}")
print(f"R^2 value is {r_squared}")
#Loop through the data and print out the predicted prices and the 
predict = model.predict(xtest)
predict = np.around(predict, 2)
print(predict)
#actual prices
for index in range(len(xtest)):
    actual = ytest[index]
    predicted_y = predict[index]
    x_coord = xtest[index]
    print(f"CO AQI Value: {x_coord[2]} Ozone AQI Value: {x_coord[3]} NO2 AQI Value: {x_coord[3]} PM2.5 AQI Value: {x_coord[4]} Actual: {actual} Predicted: {predicted_y}")