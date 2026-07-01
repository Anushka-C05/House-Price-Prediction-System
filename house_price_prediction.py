import pandas as pd

data = pd.read_csv("house_price.csv")

print(data.head())
print(data.shape)

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

predictions = model.predict(X_test)
print(predictions[:5])

from sklearn.metrics import r2_score
score = r2_score(y_test, predictions)
print("R² Score:", score)

# A score closer to 1 indicates better predictions.

house = [[1500, 3, 2]]
price = model.predict(house)
print("Predicted House Price:", price[0])
# Example:
# Predicted House Price: 385000

import matplotlib.pyplot as plt
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()