from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import phe as paillier

# Initialize the API
app = FastAPI(title="SecureMed-AI HE Backend")

# 1. FIX CORS: Allow everything so the browser stops blocking it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# 2. Load the ML Models
with open("diabetes_svm_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

with open("heart_svm_model.pkl", "rb") as f:
    heart_model = pickle.load(f)

# 3. FIX PYDANTIC: Accept large numbers as strings
class EncryptedPatientData(BaseModel):
    public_key_n: str 
    encrypted_features: list[tuple[str, int]] 

@app.post("/api/predict/diabetes")
async def predict_diabetes(data: EncryptedPatientData):
    try:
        public_key = paillier.PaillierPublicKey(n=int(data.public_key_n))
        enc_features = [paillier.EncryptedNumber(public_key, int(c[0]), int(c[1])) for c in data.encrypted_features]
        
        # CRITICAL FIX: Convert numpy.float64 to standard Python float to prevent phe library crashes
        weights = [float(w) for w in diabetes_model.coef_[0]]
        intercept = float(diabetes_model.intercept_[0])
        
        encrypted_prediction = sum([w * x for w, x in zip(weights, enc_features)]) + intercept
        
        return {
            "status": "success",
            "encrypted_result": (str(encrypted_prediction.ciphertext()), encrypted_prediction.exponent)
        }
    except Exception as e:
        # If Python fails, send the exact reason back to React
        return {"status": "error", "message": str(e)}

@app.post("/api/predict/heart")
async def predict_heart(data: EncryptedPatientData):
    try:
        public_key = paillier.PaillierPublicKey(n=int(data.public_key_n))
        enc_features = [paillier.EncryptedNumber(public_key, int(c[0]), int(c[1])) for c in data.encrypted_features]
        
        # CRITICAL FIX: Convert numpy.float64 to standard Python float
        weights = [float(w) for w in heart_model.coef_[0]]
        intercept = float(heart_model.intercept_[0])
        
        encrypted_prediction = sum([w * x for w, x in zip(weights, enc_features)]) + intercept
        
        return {
            "status": "success",
            "encrypted_result": (str(encrypted_prediction.ciphertext()), encrypted_prediction.exponent)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}