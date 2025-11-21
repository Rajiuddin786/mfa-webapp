import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import pickle
import os

def extract_features(df):

    dwell_time = df['dwell_time'].dropna()
    flight_time = df['flight_time'].dropna()
    press_press=df['press_press'].dropna()
    release_release=df['release_realease'].dropna()
    
    features = {
        'dwell_mean': dwell_time.mean(),
        'dwell_std': dwell_time.std(),
        'dwell_median': dwell_time.median(),
        'dwell_min': dwell_time.min(),
        'dwell_max': dwell_time.max(),
        'dwell_q25': dwell_time.quantile(0.25),
        'dwell_q75': dwell_time.quantile(0.75),
        'flight_mean': flight_time.mean(),
        'flight_std': flight_time.std(),
        'flight_median': flight_time.median(),
        'flight_min': flight_time.min(),
        'flight_max': flight_time.max(),
        'flight_q25': flight_time.quantile(0.25),
        'flight_q75': flight_time.quantile(0.75),
        'press_press_mean': press_press.mean(),
        'press_press_std': press_press.std(),
        'press_press_median': press_press.median(),
        'press_press_min': press_press.min(),
        'press_press_max': press_press.max(),
        'press_press_q25': press_press.quantile(0.25),
        'press_press_q75': press_press.quantile(0.75),
        'release_release_mean': release_release.mean(),
        'release_release_std': release_release.std(),
        'release_release_median': release_release.median(),
        'release_release_min': release_release.min(),
        'release_release_max': release_release.max(),
        'release_release_q25': release_release.quantile(0.25),
        'release_release_q75': release_release.quantile(0.75),
    }
    
    return features


csv_files = ['data/sample1.csv', 'data/sample2.csv', 'data/sample3.csv', 'data/sample4.csv', 'data/sample5.csv']
attempts = []
features_list = []

for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()
    
    attempts.append(df) 

    features = extract_features(df)
    features_list.append(features)


X = pd.DataFrame(features_list).values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



model = IsolationForest(
    contamination=0.2, 
    random_state=42,
    n_estimators=60,
    bootstrap=False
)

model.fit(X_scaled)

avg_length=sum([len(att) for att in attempts])/len(attempts)

model_data = {
    'model': model,
    'scaler': scaler,
    'password_length': int(avg_length),
    'num_training_attempts': len(attempts),
}
i=1
while(os.path.exists(f"models/model{i}.pkl")):
    i+=1
with open(f"models/model{i}.pkl", 'wb') as f:
    pickle.dump(model_data, f)



feature_df = pd.DataFrame(features_list)
stats_df = feature_df[['dwell_mean', 'dwell_std', 'flight_mean', 'flight_std']].describe()
print(stats_df.round(2))