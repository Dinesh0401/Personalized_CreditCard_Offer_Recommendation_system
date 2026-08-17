
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def engineer_flagship_features(df):
    df = df.copy()
    categories = ['travel', 'dining', 'online_shopping', 'fuel']
    for cat in categories:
        df[f'{cat}_affinity'] = df[f'{cat}_spend'] / (df['monthly_total_spend'] + 1)
    
    df['income_to_spend_ratio'] = df['monthly_income'] / (df['monthly_total_spend'] + 1)
    
    le_emp = LabelEncoder()
    df['employment_enc'] = le_emp.fit_transform(df['employment_status'])
    
    le_pref = LabelEncoder()
    df['pref_reward_enc'] = le_pref.fit_transform(df['preferred_reward'])
    
    return df, le_emp, le_pref

def simulate_acceptance(row):
    score = 0
    if row['credit_score'] > 700: score += 0.4
    if row['income_to_spend_ratio'] > 2: score += 0.3
    if row['age'] > 30 and row['age'] < 60: score += 0.2
    prob = np.clip(score + np.random.normal(0, 0.1), 0, 1)
    return 1 if prob > 0.5 else 0
