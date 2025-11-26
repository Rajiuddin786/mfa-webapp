import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("models/model1.h5",compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("threshold.pkl", "rb") as f:
    threshold = pickle.load(f)


def predict_user(df, scaler, model, seq_len=8, threshold=0.01):

    # select required features
    df = df[['dwell_time','flight_time','press_press','release_release']].dropna()
    data_scaled = scaler.transform(df)

    # pad if too short
    if len(data_scaled) < seq_len:
        pad_len = seq_len - len(data_scaled)
        padding = np.zeros((pad_len, data_scaled.shape[1]))
        data_scaled = np.vstack([padding, data_scaled])

    # create sequences
    sequences = []
    for i in range(len(data_scaled) - seq_len + 1):
        sequences.append(data_scaled[i:i+seq_len])

    sequences = np.array(sequences)

    # model prediction
    recon = model.predict(sequences, verbose=0)

    # reconstruction error
    mse = np.mean(np.power(sequences - recon, 2), axis=(1,2))
    avg_mse = mse.mean()

    return ("ACCEPT" if avg_mse <= threshold else "REJECT", avg_mse)


new_df = pd.read_csv('data/sample19.csv')
result, mse_score = predict_user(new_df, scaler, model, threshold=threshold)

error_percent = mse_score * 100
match_percent = (1 - mse_score) * 100

print(f"Result: {result}")
print(f"Reconstruction Error: {error_percent:.2f}%")
print(f"Match Score: {match_percent:.2f}%")

