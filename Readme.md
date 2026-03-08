
## Description
- **keras-classification-task.ipynb**: contains data preprocessing, model training, early stopping, dropout, and evaluation.
- **deploy.py**: FastAPI application used to serve predictions.
- **model_weights.pkl**: saved trained model weights.
- **scaler_weights.pkl**: saved scaler used for feature preprocessing.

## Running the API

Run the FastAPI application:

```bash
python -m uvicorn deploy:app --reload

Open the API documentation:

http://127.0.0.1:8000/docs
Prediction Endpoint

POST /predict

Input example:

{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.0787, 1.095, 0.9053, 8.589, 153.4, 0.0064, 0.049, 0.0537, 0.0159, 0.03, 0.0062, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}

Output example:

{
  "prediction": 0,
  "probability": 0.0011
}
Notes

The model expects 30 input features.

The scaler is applied before making predictions to ensure consistency with the training process.