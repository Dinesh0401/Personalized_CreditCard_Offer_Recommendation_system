
import pandas as pd

def get_eligible_candidates(customer_data, cards_df):
    # Stage 1: Eligibility Filtering
    annual_income = customer_data['monthly_income'] * 12
    eligible = cards_df[
        (cards_df['min_income_req'] <= annual_income) &
        (cards_df['min_credit_score'] <= customer_data['credit_score'])
    ].copy()
    return eligible
