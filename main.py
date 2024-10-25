import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r"C:\Users\saleh\Downloads\Nairobi Office Price Ex.csv")

# Extract the relevant columns (assuming column names are "SIZE" for office size and "PRICE" for office price)
x = df['SIZE'].values  # Lowercase to avoid shadowing
y = df['PRICE'].values

# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(x_vals, y_vals, slope, intercept, lr, n_epochs):
    n = len(y_vals)
    for epoch in range(n_epochs):
        y_predicted = slope * x_vals + intercept
        error = mean_squared_error(y_vals, y_predicted)
        print(f'Epoch {epoch + 1}, MSE: {error:.4f}')

        # Calculate gradients
        gradient_m = (-2/n) * np.dot(x_vals, (y_vals - y_predicted))
        gradient_c = (-2/n) * np.sum(y_vals - y_predicted)

        # Update parameters
        slope -= lr * gradient_m
        intercept -= lr * gradient_c

    return slope, intercept

# Initialize parameters
np.random.seed(42)  # For reproducibility
slope = np.random.rand()  # Random initial slope
intercept = np.random.rand()  # Random initial y-intercept
learning_rate = 0.01  # Learning rate
epochs = 10  # Number of epochs

# Train the model
slope, intercept = gradient_descent(x, y, slope, intercept, learning_rate, epochs)

# Final predictions and saving to DataFrame
df['predicted_price'] = slope * x + intercept

# Plotting the line of best fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", label="Data Points")
plt.plot(x, df['predicted_price'], color="red", label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft)")
plt.ylabel("Office Price")
plt.title("Office Price vs. Size with Line of Best Fit")
plt.legend()
plt.show()

# Save DataFrame with predictions to a CSV file
df.to_csv(r"C:\Users\saleh\Downloads\predicted_office_prices.csv", index=False)
print("Predictions saved to 'predicted_office_prices.csv'")

# Predicting office price for size 100 sq. ft.
office_size = 100
predicted_price = slope * office_size + intercept
print(f"The predicted price for an office size of 100 sq. ft. is: {predicted_price:.2f}")
