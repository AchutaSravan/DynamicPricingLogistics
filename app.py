from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("dynamic_pricing_model.pkl")

# FastAPI app
app = FastAPI(title="Dynamic Logistics Pricing API")

# Input schema
class PricingRequest(BaseModel):
    origin: str
    destination: str
    route_distance_km: float
    order_urgency: str
    pickup_time_hour: int
    pickup_day_of_week: int
    market_demand_index: float
    competitor_price: float
    fuel_index: float
    base_cost: float
    historical_acceptance_rate: float

@app.post("/predict_price")
def predict_price(request: PricingRequest):
    try:
        input_data = request.dict()

        # Manual encoding
        origin_map = {'PHX': 0, 'LAX': 1, 'NYC': 2, 'ATL': 3, 'CHI': 4}
        destination_map = {'SEA': 0, 'DAL': 1, 'MIA': 2, 'DEN': 3, 'BOS': 4}
        urgency_map = {'Low': 0, 'Medium': 1, 'High': 2}

        input_data['origin'] = origin_map.get(input_data['origin'], -1)
        input_data['destination'] = destination_map.get(input_data['destination'], -1)
        input_data['order_urgency'] = urgency_map.get(input_data['order_urgency'], -1)

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        return {
            "predicted_price": round(prediction, 2),
            "model_version": "v1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

