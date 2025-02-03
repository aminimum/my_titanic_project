import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Titanic Survival Prediction API")

# Load the trained model
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)


class Passenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int


@app.get("/")
def read_root():
    return {"message": "Welcome to Titanic Survival Prediction API"}


@app.post("/predict")
def predict(passenger: Passenger):
    data = passenger.dict()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"Survived": int(prediction)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
