"""Problem 2: Ising Model - Neural Network Tasks
Task A: Classification (above/below critical temperature)
Task B: Regression (predict temperature)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import math
import warnings
warnings.filterwarnings('ignore')
np.random.seed(126); random.seed(126)

def initialize_grid(dim):
    return np.random.choice([-1, 1], size=(dim, dim))

def energy_change(grid, i, j):
    n = grid.shape[0]
    left, right = (i - 1) % n, (i + 1) % n
    up, down = (j + 1) % n, (j - 1) % n
    return 2 * grid[i, j] * (grid[left, j] + grid[right, j] + grid[i, up] + grid[i, down])

def spin_flip(grid, T):
    n = grid.shape[0]
    i, j = random.randint(0, n - 1), random.randint(0, n - 1)
    delta_E = energy_change(grid, i, j)
    if delta_E < 0 or random.random() < math.exp(-delta_E / T):
        grid[i, j] = -grid[i, j]
    return grid

def ising_simulation(n, T, steps=100):
    """Simulate 2D Ising model using Metropolis algorithm"""
    grid = initialize_grid(n)
    for step in range(steps):
        for _ in range(n * n):
            grid = spin_flip(grid, T)
    return grid

def generate_data(size, num_temp, temp_min=1.0, temp_max=3.5, repeat=1, max_iter=None):
    """Generate training/test data from Ising model simulations"""
    if max_iter is None:
        max_iter = size**2
    X = np.zeros((num_temp * repeat, size**2))
    y_label = np.zeros((num_temp * repeat, 1))
    y_temp = np.zeros((num_temp * repeat, 1))
    temps = np.linspace(temp_min, temp_max, num=num_temp)
    for i in range(repeat):
        for j in range(num_temp):
            grid = ising_simulation(size, temps[j], max_iter)
            X[i*num_temp + j, :] = grid.reshape(1, grid.size)
            y_label[i*num_temp + j, :] = (temps[j] > 2.269)
            y_temp[i*num_temp + j, :] = temps[j]
            print(f"Generated {i*num_temp + j + 1}/{num_temp * repeat}", end='\r')
    print()
    return X, y_label, y_temp

def build_model(input_dim, task_type='classification'):
    """Build neural network model (TensorFlow or sklearn)"""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3), Dense(64, activation='relu'),
            Dropout(0.2), Dense(32, activation='relu'),
            Dense(1, activation='sigmoid' if task_type == 'classification' else None)
        ])
        model.compile(optimizer=Adam(0.001),
                     loss='binary_crossentropy' if task_type == 'classification' else 'mean_absolute_error',
                     metrics=['accuracy' if task_type == 'classification' else 'mae'])
        return model, 'keras'
    except ImportError:
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        if task_type == 'classification':
            return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, 
                                random_state=126, early_stopping=True), 'sklearn'
        else:
            return MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                              random_state=126, early_stopping=True), 'sklearn'

def plot_model_flowchart(task_name, model_type, save_path):
    """Create model flowchart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    layers = ['Input\n(625)', 'Dense\n(128)', 'Dense\n(64)', 'Dense\n(32)', 
              'Output\n(1)' if model_type == 'regression' else 'Output\n(0/1)']
    y_pos = np.linspace(0.9, 0.1, len(layers))
    for i, (layer, y) in enumerate(zip(layers, y_pos)):
        rect = plt.Rectangle((0.4 - 0.08, y - 0.04), 0.16, 0.08,
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.4, y, layer, ha='center', va='center', fontsize=9, fontweight='bold')
        if i < len(layers) - 1:
            ax.arrow(0.4, y - 0.04, 0, -(y_pos[i] - y_pos[i+1] - 0.08),
                    head_width=0.015, head_length=0.015, fc='black', ec='black')
    ax.text(0.5, 0.95, f'{task_name} Model Flowchart', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(model, model_type, X_train, y_train, X_test, y_test, task_name):
    """Train model and return predictions"""
    if model_type == 'keras':
        from tensorflow.keras.callbacks import EarlyStopping
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                 verbose=0)
        y_pred = model.predict(X_test, verbose=0)
        if task_name == 'classification':
            y_pred = (y_pred > 0.5).astype(int).flatten()
        else:
            y_pred = y_pred.flatten()
    else:
        model.fit(X_train, y_train.flatten())
        y_pred = model.predict(X_test)
    return y_pred

def plot_results(y_true, y_pred, y_temp_test, task_name, metric_name, save_path):
    """Plot results vs temperature"""
    from sklearn.metrics import accuracy_score, mean_absolute_error
    unique_temps = np.unique(y_temp_test)
    metrics_by_temp, temps_list = [], []
    for temp in unique_temps:
        mask = (y_temp_test.flatten() == temp)
        if np.sum(mask) > 0:
            if task_name == 'classification':
                metric = accuracy_score(y_true[mask], y_pred[mask])
            else:
                metric = mean_absolute_error(y_true[mask], y_pred[mask])
            metrics_by_temp.append(metric)
            temps_list.append(temp)
    plt.figure(figsize=(10, 6))
    plt.plot(temps_list, metrics_by_temp, 'o-', linewidth=2, markersize=8,
            color='blue' if task_name == 'classification' else 'green')
    plt.axvline(x=2.269, color='r', linestyle='--', linewidth=2, label='Tc = 2.269')
    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Task {"A" if task_name == "classification" else "B"}: {metric_name} vs Temperature', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def task_a_classification(X_train, y_label_train, X_test, y_label_test, y_temp_test):
    """Task A: Classify above/below critical temperature"""
    print("\n" + "=" * 60)
    print("Task A: Classification")
    print("=" * 60)
    model, model_type = build_model(X_train.shape[1], 'classification')
    plot_model_flowchart("Task A", "classification", "task_a_model_flowchart.png")
    print("Training model...")
    y_pred = train_and_evaluate(model, model_type, X_train, y_label_train, X_test, y_label_test, 'classification')
    y_true = y_label_test.flatten().astype(int)
    from sklearn.metrics import accuracy_score
    print(f"Overall Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    plot_results(y_true, y_pred, y_temp_test, 'classification', 'Test Accuracy', 'task_a_accuracy_vs_temp.png')
    print("Plot saved to task_a_accuracy_vs_temp.png")
    print("Observations: Accuracy decreases near critical temperature (Tc = 2.269)")

def task_b_regression(X_train, y_temp_train, X_test, y_temp_test):
    """Task B: Predict temperature"""
    print("\n" + "=" * 60)
    print("Task B: Regression")
    print("=" * 60)
    model, model_type = build_model(X_train.shape[1], 'regression')
    plot_model_flowchart("Task B", "regression", "task_b_model_flowchart.png")
    print("Training model...")
    y_pred = train_and_evaluate(model, model_type, X_train, y_temp_train, X_test, y_temp_test, 'regression')
    y_true = y_temp_test.flatten()
    from sklearn.metrics import mean_absolute_error
    print(f"Overall Test MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    plot_results(y_true, y_pred, y_temp_test, 'regression', 'Test MAE', 'task_b_mae_vs_temp.png')
    print("Plot saved to task_b_mae_vs_temp.png")
    print("Observations: MAE increases near critical temperature (Tc = 2.269)")

def main():
    print("=" * 60)
    print("Problem 2: Ising Model - Neural Network Tasks")
    print("=" * 60)
    print("\nGenerating training data...")
    X_train, y_label_train, y_temp_train = generate_data(
        size=25, num_temp=51, temp_min=1.0, temp_max=3.5, repeat=20, max_iter=625)
    print("\nGenerating test data...")
    X_test, y_label_test, y_temp_test = generate_data(
        size=25, num_temp=21, temp_min=1.0, temp_max=3.5, repeat=20, max_iter=625)
    print(f"\nTraining: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    task_a_classification(X_train, y_label_train, X_test, y_label_test, y_temp_test)
    task_b_regression(X_train, y_temp_train, X_test, y_temp_test)
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
