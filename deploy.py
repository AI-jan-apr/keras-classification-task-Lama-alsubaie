from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

app = FastAPI()

# Load scaler
with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load model weights
with open("model_weights.pkl", "rb") as f:
    weights = pickle.load(f)

# Rebuild the same model architecture
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Build model before setting weights
model.build((None, 30))
model.set_weights(weights)

class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Keras classification model is running"}

@app.post("/predict")
def predict(data: PredictionRequest):
    if len(data.features) != 30:
        return {"error": "You must provide exactly 30 features."}

    features = np.array(data.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0][0]
    predicted_class = int(prediction > 0.5)

    return {
        "prediction": predicted_class,
        "probability": float(prediction)
    }