# -*- coding: utf-8 -*-
"""Problem 1: Time Series Forecasting for Energy Consumption Data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
import os
import pickle
warnings.filterwarnings('ignore')
np.random.seed(126)

def load_data_with_date(data_path, train_end, pred_start, pred_end):
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    target_col = [c for c in df.columns if 'MW' in c.upper()][0]
    train_data = df[df.index <= pd.to_datetime(train_end)][target_col].values
    train_data = train_data[~np.isnan(train_data)]
    pred_actual = df[(df.index >= pd.to_datetime(pred_start)) & (df.index <= pd.to_datetime(pred_end))][target_col].values
    pred_actual = pred_actual[~np.isnan(pred_actual)]
    return train_data, pred_actual

def create_sequences(data, seq_len=96, pred_len=24):
    return np.array([data[i:i+seq_len] for i in range(len(data)-seq_len-pred_len+1)]), \
           np.array([data[i+seq_len:i+seq_len+pred_len] for i in range(len(data)-seq_len-pred_len+1)])

def smape_loss(y_true, y_pred):
    import tensorflow as tf
    eps = tf.keras.backend.epsilon()
    return tf.reduce_mean(2.0 * tf.abs(y_true - y_pred) / (tf.abs(y_true) + tf.abs(y_pred) + eps))

def build_model(seq_len, pred_len, improved=True):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    if improved:
        return Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_len, 1)), Dropout(0.3),
            LSTM(128, return_sequences=True), Dropout(0.3),
            LSTM(64, return_sequences=False), Dropout(0.2),
            Dense(64, activation='relu'), Dense(32, activation='relu'), Dense(pred_len)
        ])
    else:
        return Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len, 1)), Dropout(0.2),
            LSTM(64, return_sequences=False), Dropout(0.2),
            Dense(32), Dense(pred_len)
        ])

def iterative_forecast(model, X_init, pred_hours, batch_size=24):
    predictions, current_seq = [], X_init.copy()
    for i in range(0, pred_hours, batch_size):
        pred_batch = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)[0]
        predictions.append(pred_batch if i + batch_size <= pred_hours else pred_batch[:pred_hours - i])
        if i + batch_size <= pred_hours:
            current_seq = np.concatenate([current_seq[batch_size:], pred_batch])
    return np.concatenate(predictions)

def calculate_smape(y_true, y_pred):
    """Calculate sMAPE value"""
    eps = 1e-8
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps))

def plot_model_architecture():
    try:
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        boxes = [('Raw Data', 1, 8), ('Normalization', 3.5, 8), ('Sequence Creation', 6, 8), ('LSTM Model', 3.5, 6),
                 ('Training', 1, 4), ('Iterative Forecast', 3.5, 4), ('Denormalization', 6, 4), ('Evaluation', 3.5, 2)]
        for text, x, y in boxes:
            ax.add_patch(FancyBboxPatch((x-0.75, y-0.4), 1.5, 0.8, boxstyle="round,pad=0.1",
                                       edgecolor='black', facecolor='lightblue', linewidth=2))
            ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
        arrows = [(1, 8, 3.5, 8), (3.5, 8, 6, 8), (6, 7.6, 3.5, 6.4), (3.5, 5.6, 1, 4.4),
                  (1, 4, 3.5, 4), (3.5, 4, 6, 4), (6, 3.6, 3.5, 2.4)]
        for x1, y1, x2, y2 in arrows:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=20,
                                       linewidth=2, color='darkblue'))
        ax.set_title('Neural Network Model Architecture', fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Model architecture diagram saved to model_architecture.png")
    except Exception as e:
        print(f"Error plotting model architecture: {e}")

def forecast(data_path, train_end, pred_start, pred_end, seq_len=96, problem_name='', retrain=False):
    print(f"\n{'='*60}\n{problem_name}\n{'='*60}")
    train_data, pred_actual = load_data_with_date(data_path, train_end, pred_start, pred_end)
    print(f"Training: {len(train_data)} points, Prediction: {len(pred_actual)} hours")
    
    # Generate model and scaler file paths
    model_name = 'model_problem_a.h5' if 'A' in problem_name else 'model_problem_b.h5'
    scaler_name = 'scaler_problem_a.pkl' if 'A' in problem_name else 'scaler_problem_b.pkl'
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    pred_len = 24
    
    X_train, y_train = create_sequences(train_scaled, seq_len=seq_len, pred_len=pred_len)
    X_train_r = np.array(X_train, dtype=np.float32).reshape((len(X_train), seq_len, 1))
    y_train_all = np.array(y_train, dtype=np.float32)
    
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        model = build_model(seq_len, pred_len, improved=True)
        model.compile(optimizer='adam', loss=smape_loss, metrics=['mae'])
        
        # Check if saved model exists and not forcing retrain
        if not retrain and os.path.exists(model_name) and os.path.exists(scaler_name):
            print(f"Loading saved model from {model_name}...")
            model.load_weights(model_name)
            with open(scaler_name, 'rb') as f:
                scaler = pickle.load(f)
            train_scaled = scaler.transform(train_data.reshape(-1, 1)).flatten()
            print("Model and scaler loaded successfully!")
        else:
            if retrain:
                print("Force retraining model (--retrain flag set)...")
            else:
                print("Training new model...")
            checkpoint = ModelCheckpoint(model_name, monitor='loss', save_best_only=True, 
                                       save_weights_only=True, verbose=0)
            model.fit(X_train_r, y_train_all, epochs=100, batch_size=64,
                     callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                               checkpoint], verbose=0)
            # Save scaler
            with open(scaler_name, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Model saved to {model_name}, scaler saved to {scaler_name}")
        
        # Use the last seq_len points of training data as initial sequence
        long_term_pred = iterative_forecast(model, train_scaled[-seq_len:], len(pred_actual))[:len(pred_actual)]
        long_term_pred_actual = scaler.inverse_transform(long_term_pred.reshape(-1, 1)).flatten()
        pred_actual_vals = scaler.inverse_transform(pred_actual.reshape(-1, 1)).flatten()
        
        # Calculate sMAPE
        smape = calculate_smape(pred_actual_vals, long_term_pred_actual)
        print(f"\nForecasting Results - SMAPE: {smape:.4f}")
        
        plot_len = min(2000, len(pred_actual_vals))
        time_idx = pd.date_range(start=pd.to_datetime(pred_start), periods=plot_len, freq='H')
        plt.figure(figsize=(16, 6))
        plt.plot(time_idx, pred_actual_vals[:plot_len], label='Actual', linewidth=1, alpha=0.7)
        plt.plot(time_idx, long_term_pred_actual[:plot_len], label='Predicted', linewidth=1, linestyle='--', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption (MW)')
        plt.title(f'{problem_name} (First {plot_len} Hours)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = 'problem_a_forecast.png' if 'A' in problem_name else 'long_term_forecast.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {filename}")
        return smape
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    import sys
    plot_model_architecture()
    
    retrain = '--retrain' in sys.argv
    print("\n" + "="*60 + "\nRunning both Problem A and Problem B\n" + "="*60)
    
    smape_a = forecast('EnergyConsumption_hourly.csv', '2017-12-31', '2018-01-01', '2018-08-31',
                      seq_len=96, problem_name='Problem A: Forecasting (2018-01 to 2018-08)', retrain=retrain)
    smape_b = forecast('EnergyConsumption_hourly.csv', '2016-12-31', '2017-01-01', '2018-08-31',
                      seq_len=168, problem_name='Problem B: Long-Term Forecasting (2017-01 to 2018-08)', retrain=retrain)
    
    print("\n" + "="*60 + "\nSummary:")
    if smape_a is not None:
        print(f"  Problem A (2018-01 to 2018-08) - SMAPE: {smape_a:.4f}")
    if smape_b is not None:
        print(f"  Problem B (2017-01 to 2018-08) - SMAPE: {smape_b:.4f}")
    print("="*60)
