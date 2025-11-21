import pandas as pd
import numpy as np
import pickle

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


with open('models/model1.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
expected_length = model_data['password_length']

print("Model loaded successfully")




df = pd.read_csv('data/sample6.csv')
df.columns = df.columns.str.strip().str.lower()

print(f"Loaded {len(df)} keystrokes for verification")


if abs(len(df) - expected_length) > 2: 
    print(f"WARNING: Expected {expected_length} keystrokes, got {len(df)}")
    print("This might be a different password!")


features = extract_features(df)
X = pd.DataFrame([features]).values


X_scaled = scaler.transform(X)


prediction = model.predict(X_scaled)[0]
score = model.decision_function(X_scaled)[0]


is_authentic = (prediction == 1)
confidence = round((score + 0.5) * 100, 2)
confidence = max(0, min(100, confidence)) 




if is_authentic:
    print("AUTHENTICATION SUCCESSFUL")
    print("Status: VERIFIED ")
    print(f"Confidence: {confidence}%")
    print(f"Decision Score: {score:.4f}")
    print("\n➜ This typing pattern matches the enrolled user!")
else:
    print("AUTHENTICATION FAILED")
    print("Status: REJECTED ")
    print(f"Confidence: {confidence}%")
    print(f"Decision Score: {score:.4f}")





print("\nVerification Sample Statistics:")
print(f"  Avg Dwell Time: {features['dwell_mean']:.2f} ms")
print(f"  Avg Flight Time: {features['flight_mean']:.2f} ms")


print("\nSecurity Level:")
if is_authentic and confidence > 70:
    print("   HIGH - Strong match")
elif is_authentic and confidence > 50:
    print("   MEDIUM - Acceptable match")
elif is_authentic:
    print("   LOW - Weak match, consider re-enrollment")
else:
    print("   REJECTED - No match")
