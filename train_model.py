import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,RepeatVector
from tensorflow.keras.optimizers import Adam
import os

i=1
csv_files=[]
while os.path.exists(f"data/sample{i}.csv"):
    csv_files.append(f'data/sample{i}.csv')
    i+=1

all_sequences = []
seq_len = 8
scaler = MinMaxScaler()

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()

    # Only raw signals, row wise
    df = df[['dwell_time','flight_time','press_press','release_release']].dropna()

    # Scale
    data_scaled = scaler.fit_transform(df)

    # Create sequences
    def create_sequences(data, seq_len):
        seqs = []
        for i in range(len(data) - seq_len + 1):
            seqs.append(data[i:i+seq_len])
        return np.array(seqs)

    seqs = create_sequences(data_scaled, seq_len)
    all_sequences.append(seqs)

# Combine all training sequences
X = np.vstack(all_sequences)

print("Final Shape:", X.shape)

timesteps=X.shape[1]
features=X.shape[2]
latent_dim=16

inputs=Input(shape=(timesteps,features))
encoded=LSTM(latent_dim)(inputs)

decoded=RepeatVector(timesteps)(encoded)
decoded=LSTM(features,return_sequences=True)(decoded)

autoencoder=Model(inputs,decoded)
autoencoder.compile(optimizer=Adam(0.001),loss='mse')

autoencoder.summary()

history=autoencoder.fit(X,X,epochs=30,batch_size=32,validation_split=0.2,verbose=1)
recon=autoencoder.predict(X)
autoencoder.save("models/model1.h5")

mse = np.mean(np.power(X - recon, 2), axis=(1,2))
threshold = np.percentile(mse, 95)  

print("Threshold:", threshold)
import pickle
with open("threshold.pkl", "wb") as f:
    pickle.dump(threshold, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
