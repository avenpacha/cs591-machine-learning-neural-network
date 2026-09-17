import numpy as np
import matplotlib.pyplot as plt
import os

# Create a directory to save plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load Data
data = np.loadtxt('Part1_x_y_Values.txt', skiprows=1)
X = data[:, 0]
Y = data[:, 1]

# Normalize X to [0, 1] to prevent exploding gradients with higher powers
X_min, X_max = X.min(), X.max()
X_norm = (X - X_min) / (X_max - X_min)

# Helper function to create polynomial features
def create_features(x, degree):
    # If degree=2 (3 features): [1, x, x^2]
    # If degree=3 (4 features): [1, x, x^2, x^3]
    return np.column_stack([x**i for i in range(degree + 1)])

# Gradient Descent Implementation
def gradient_descent(X_b, Y, lr, epochs=100):
    m = len(Y)
    weights = np.zeros(X_b.shape[1])
    weight_history = []
    
    for epoch in range(epochs):
        predictions = X_b.dot(weights)
        errors = predictions - Y
        gradients = (2/m) * X_b.T.dot(errors)
        weights = weights - lr * gradients
        weight_history.append(weights.copy())
        
    return weight_history

# Stochastic Gradient Descent Implementation
def stochastic_gradient_descent(X_b, Y, lr, epochs=100):
    m = len(Y)
    weights = np.zeros(X_b.shape[1])
    weight_history = []
    
    for epoch in range(epochs):
        # Shuffle data for SGD
        indices = np.random.permutation(m)
        X_b_shuffled = X_b[indices]
        Y_shuffled = Y[indices]
        
        for i in range(m):
            xi = X_b_shuffled[i:i+1]
            yi = Y_shuffled[i:i+1]
            prediction = xi.dot(weights)
            error = prediction - yi
            gradients = 2 * xi.T.dot(error)
            weights = weights - lr * gradients.flatten()
            
        weight_history.append(weights.copy())
        
    return weight_history

# Plotting function to show ALL epochs (Bonus Points)
def plot_training(X_orig, Y, X_b, weight_history, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_orig, Y, color='blue', label='Data Points', alpha=0.5)
    
    # Plot curve for each epoch with increasing opacity/color to show learning over time
    epochs = len(weight_history)
    X_plot = np.linspace(X_orig.min(), X_orig.max(), 100)
    X_plot_norm = (X_plot - X_min) / (X_max - X_min)
    X_plot_b = np.column_stack([X_plot_norm**i for i in range(X_b.shape[1])])
    
    for i, w in enumerate(weight_history):
        alpha = (i + 1) / epochs # Later epochs are darker
        Y_plot = X_plot_b.dot(w)
        if i == epochs - 1:
            plt.plot(X_plot, Y_plot, color='red', linewidth=2, label='Final Epoch')
        elif i % 10 == 0: # Plot every 10th epoch to avoid clutter, but it represents the history
            plt.plot(X_plot, Y_plot, color='gray', alpha=alpha*0.5)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{filename}.png')
    plt.close()

# --- Run Experiments for Report ---
degrees = [2, 3, 4] # 3 features, 4 features, 5 features (Bonus 1)
learning_rates = [0.01, 0.05, 0.1] # Base LR, and two others (Bonus 2)
epochs = 100

for degree in degrees:
    X_poly = create_features(X_norm, degree)
    feature_count = degree + 1
    
    for lr in learning_rates:
        # Run GD
        print(f"Running GD: {feature_count} features, LR={lr}")
        gd_history = gradient_descent(X_poly, Y, lr, epochs)
        plot_training(X, Y, X_poly, gd_history, 
                      f'GD: {feature_count} Features, LR={lr}', 
                      f'GD_{feature_count}feat_lr{lr}')
        
        # Run SGD
        print(f"Running SGD: {feature_count} features, LR={lr}")
        sgd_history = stochastic_gradient_descent(X_poly, Y, lr, epochs)
        plot_training(X, Y, X_poly, sgd_history, 
                      f'SGD: {feature_count} Features, LR={lr}', 
                      f'SGD_{feature_count}feat_lr{lr}')

print("All training complete. Check the 'plots' folder for your graphs!")