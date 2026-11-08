import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv("advertising.csv")

print(df.head())
print(df.shape)

print(df.info())

print(df.describe())

print(df.isnull().sum())
df = df.drop_duplicates()

df = df.dropna()

print("Cleaned Dataset:")
print(df.head())
X = df[['TV', 'Radio', 'Newspaper']]

y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)

print("Model Training Completed!")
y_pred = model.predict(X_test)

print("Actual Sales:")
print(y_test.values)

print("Predicted Sales:")
print(y_pred)
mae = mean_absolute_error(y_test, y_pred)

accuracy = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)

print("R2 Score:", accuracy)
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()
new_data = pd.DataFrame({
    'TV': [200],
    'Radio': [40],
    'Newspaper': [30]
})

prediction = model.predict(new_data)

print("Predicted Sales:", prediction[0])
importance = pd.DataFrame({
    'Platform': X.columns,
    'Impact': model.coef_
})

print(importance)

sns.barplot(x='Platform', y='Impact', data=importance)

plt.title("Advertising Impact on Sales")

plt.show()