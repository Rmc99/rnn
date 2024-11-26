# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense
import time  # Biblioteca para medir o tempo

# Load the dataset
dataset = pd.read_csv(
    "Apple.csv", index_col="Date", parse_dates=["Date"]
).drop(["Open", "Low", "Close", "Adj Close", "Volume"], axis=1)

# Print dataset information
print(dataset.head())
print(dataset.describe())
print(dataset.isna().sum())

# Define training and testing period
tstart = "1980-12-12"
tend = "2019-12-31"

# Function to plot training and testing data
def train_test_plot(dataset, tstart, tend):
    dataset.loc[tstart:tend, "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[tend:, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {tend})", f"Test ({tend} and beyond)"])
    plt.title("Apple Stock Price")
    plt.show()

train_test_plot(dataset, tstart, tend)

# Split dataset into train and test sets
def train_test_split(dataset, tstart, tend):
    train = dataset.loc[tstart:tend, "High"].values
    test = dataset.loc[tend:, "High"].values
    return train, test

training_set, test_set = train_test_split(dataset, tstart, tend)

# Scale the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set.reshape(-1, 1))
test_set_scaled = sc.transform(test_set.reshape(-1, 1))

# Function to create sequences
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        X.append(sequence[i:end_ix])
        y.append(sequence[end_ix])
    return np.array(X), np.array(y)

n_steps = 60
X_train, y_train = split_sequence(training_set_scaled, n_steps)

# Reshape for model input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Function to build models
def build_model(architecture="LSTM"):
    model = Sequential()
    if architecture == "LSTM":
        model.add(LSTM(units=64, activation="tanh", input_shape=(n_steps, 1)))
    elif architecture == "GRU":
        model.add(GRU(units=64, activation="tanh", input_shape=(n_steps, 1)))
    elif architecture == "BiLSTM":
        model.add(Bidirectional(LSTM(units=64, activation="tanh", input_shape=(n_steps, 1))))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model

# Train and evaluate models
def evaluate_model(model, X_train, y_train, test_set, n_steps, architecture):
    # Measure training time
    start_time = time.time()

    # Fit the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Tempo de treinamento para {architecture}: {training_time:.2f} segundos")

    # Prepare the test data
    inputs = dataset["High"][-len(test_set) - n_steps:].values.reshape(-1, 1)
    inputs_scaled = sc.transform(inputs)
    X_test, y_test = split_sequence(inputs_scaled, n_steps)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Make predictions
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    y_test = sc.inverse_transform(y_test)

    # Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, color="blue", label="Real Stock Price")
    plt.plot(predicted_stock_price, color="red", label="Predicted Stock Price")
    plt.title(f"Apple Stock Price Prediction ({architecture})")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    mae = mean_absolute_error(y_test, predicted_stock_price)
    r2 = r2_score(y_test, predicted_stock_price)

    print(f"Metrics for {architecture}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

# Build and evaluate models for all architectures
for arch in ["LSTM", "GRU", "BiLSTM"]:
    print(f"Evaluating {arch} model...")
    model = build_model(architecture=arch)
    evaluate_model(model, X_train, y_train, test_set, n_steps, arch)
