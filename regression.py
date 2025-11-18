import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5]).reshape(-1, 1)
plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Provided data for regression")
plt.show()
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(units=1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
history=model.fit(X, Y, epochs=100,verbose=0)
plt.plot(history.history["loss"])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Model loss")
plt.show()
predictions = model.predict(X)
plt.scatter(X, Y, label="Original data")
plt.plot(X, predictions,color="red", label="Predicted data", linewidth=2)
plt.title("Regression model prediction")
plt.legend()
plt.show()