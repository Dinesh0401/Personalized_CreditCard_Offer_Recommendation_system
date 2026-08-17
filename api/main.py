
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.recommender import PersonalizedRecommender
import uvicorn

app = FastAPI(title='Personalized Credit Card Recommendation API')

# Initialize the recommender (assumes model.joblib exists in root)
# In a real setup, this path would be configurable
try:
    recommender = PersonalizedRecommender(model_path='../model.joblib', cards_path='../data/credit_cards.csv')
except Exception as e:
    recommender = None
    print(f'Warning: Recommender not initialized: {e}')

class CustomerProfile(BaseModel):
    age: int
    monthly_income: float
    credit_score: int
    employment_status: str
    monthly_total_spend: float
    travel_spend: float
    dining_spend: float
    online_shopping_spend: float
    fuel_spend: float
    preferred_reward: str

@app.get('/health')
def health_check():
    return {'status': 'healthy'}

@app.post('/recommend')
def get_recommendation(profile: CustomerProfile):
    if not recommender:
        raise HTTPException(status_code=503, detail='Recommender service unavailable')
    
    results = recommender.recommend(profile.dict())
    return {'recommendations': results}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
