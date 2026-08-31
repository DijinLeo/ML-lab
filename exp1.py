import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["Price"] = housing.target
x = data[['AveRooms']].values
y = data['Price'].values
#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
w=0
b=0
learning_rate = 0.01
epochs = 1000
cost_history = []
n=len(x_train_scaled)
for i in range(epochs):
 y_pred =w*x_train_scaled.flatten()+b
 dw =(1/n)*np.sum((y_pred - y_train)* x_train_scaled.flatten())
 db = (1 / n) * np.sum(y_pred - y_train)
 w=w-learning_rate*dw
 b=b-learning_rate*db
 cost = (1 / (2 * n)) * np.sum((y_pred - y_train) ** 2)
 cost_history.append(cost)
 if i % 100 == 0:
 cost =(1/(2*n)) * np.sum((y_pred - y_train) **2)
 print(f"Epoch{i},Cost={cost:.4f}")
y_pred_gd=w*x_test_scaled.flatten()+b
print("Gradient Descent")
print("----------------")
print("Weight:",w)
print("Bias:",b)
print("MSE:",mean_squared_error(y_test, y_pred_gd))
print("R2 Score:",r2_score(y_test,y_pred_gd))
plt.figure(figsize=(6,4))
plt.plot(cost_history)
plt.title("Cost vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True)
plt.show()
