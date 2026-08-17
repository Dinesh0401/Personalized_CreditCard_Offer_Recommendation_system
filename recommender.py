
import pandas as pd
import joblib
from .feature_engineering import engineer_flagship_features
from .candidate_generation import get_eligible_candidates
from .retriever import CardRetriever
from .llm_explainer import generate_grounded_explanation
from .guardrails import validate_output

class PersonalizedRecommender:
    def __init__(self, model_path='model.joblib', cards_path='data/credit_cards.csv'):
        self.model = joblib.load(model_path)
        self.cards_df = pd.read_csv(cards_path)
        self.retriever = CardRetriever()

    def recommend(self, customer_profile, top_k=2):
        # 1. Eligibility Filtering
        eligible = get_eligible_candidates(customer_profile, self.cards_df)
        if eligible.empty:
            return []

        # 2. Ranking
        # Note: In production, we would map features here
        # This is a simplified call to the model
        probs = self.model.predict_proba(pd.DataFrame([customer_profile]))[0][1] # Mock indexing
        eligible['suitability_score'] = probs
        top_cards = eligible.sort_values('suitability_score', ascending=False).head(top_k)

        results = []
        for _, card in top_cards.iterrows():
            # 3. Retrieval
            rag_data = self.retriever.query(customer_profile['preferred_reward'], top_k=1)
            fact = rag_data['documents'][0][0]
            
            # 4. Explanation
            rec_bundle = card.to_dict()
            rec_bundle['fact_sheet'] = fact
            explanation = generate_grounded_explanation(rec_bundle, customer_profile)
            
            if validate_output(explanation):
                results.append({"card": card['card_name'], "explanation": explanation})
        
        return results
