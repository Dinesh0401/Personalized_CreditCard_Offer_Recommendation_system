# Personalized Credit Card Offer Recommendation System

An end-to-end **Machine Learning + Recommendation Systems + RAG + GenAI** platform that generates personalized credit-card recommendations based on customer financial profiles, spending behavior, reward preferences, and eligibility constraints.

> **Note:** This project uses **synthetic experimental data**. Model metrics and recommendation results do not represent real-world banking performance or financial advice.

## 🚀 Project Overview

The system follows a two-stage recommendation architecture:

```text
Customer Profile
       │
       ▼
Data Validation
       │
       ▼
Feature Engineering
       │
       ▼
Candidate Generation
       │
       ▼
Eligibility Filtering
       │
       ▼
ML Ranking Model
       │
       ▼
Top-K Recommendations
       │
       ▼
Hybrid RAG Retrieval
       │
       ▼
Grounded LLM Explanation
       │
       ▼
Guardrails & Validation
       │
       ▼
Personalized Recommendation
```

## ✨ Key Features

* Synthetic customer-card interaction dataset generation
* Data-quality validation and preprocessing
* Financial and spending-based feature engineering
* **Two-stage recommendation engine**

  * Stage 1: Eligibility filtering
  * Stage 2: ML-based ranking
* Multiple ML model comparison
* Top-K recommendation evaluation
* Feature importance and SHAP explainability
* ChromaDB-based semantic retrieval
* Sentence Transformer embeddings
* Hybrid retrieval using semantic similarity + structured constraints
* Grounded LLM explanations
* Financial-domain hallucination guardrails
* Recommendation trade-off analysis
* Error analysis
* Responsible AI considerations
* FastAPI-ready architecture
* Production and MLOps architecture

## 🧠 Recommendation Approach

### Stage 1 — Candidate Generation & Eligibility

Cards that violate known hard requirements are removed before ranking.

Examples:

* Minimum income
* Minimum credit score
* Eligibility restrictions
* Customer fee preference

```text
All Cards
   ↓
Eligibility Rules
   ↓
Eligible Candidates
```

### Stage 2 — ML Ranking

The eligible cards are transformed into customer-card interaction features and scored by the ranking model.

```text
Customer + Card
      ↓
Interaction Features
      ↓
ML Model
      ↓
Suitability Score
      ↓
Ranking
      ↓
Top-K Cards
```

This prevents the ML model from recommending a card that already fails a known hard eligibility requirement.

## 📊 Feature Engineering

Important customer-level features include:

* Income-to-spending ratio
* Debt burden proxy
* Travel affinity
* Dining affinity
* Fuel affinity
* Shopping affinity
* Grocery affinity
* Spending diversity
* Financial stability score
* Creditworthiness score
* Fee affordability score
* Reward preference score

Customer-card interaction features include:

* Reward match score
* Travel match
* Dining match
* Fuel match
* Shopping match
* Fee affordability
* Income eligibility
* Credit-score eligibility
* Benefit match score
* Overall card affinity

## 🤖 Machine Learning

The project compares multiple approaches rather than assuming Random Forest is automatically the best model.

Models include:

* Heuristic / popularity baseline
* Logistic Regression
* Random Forest
* Gradient Boosting / HistGradientBoosting

Recommendation performance is evaluated using ranking-oriented metrics such as:

* Precision@K
* Recall@K
* Hit Rate@K
* NDCG@K
* MAP@K where applicable

Classification metrics such as ROC-AUC and PR-AUC are used only for appropriate binary prediction components.

## 🔎 RAG Knowledge Base

Credit-card information is stored in a ChromaDB vector database using Sentence Transformer embeddings.

Each card document contains structured information such as:

* Card name
* Issuer
* Annual fee
* Minimum income
* Minimum credit score
* Rewards
* Benefits
* Restrictions
* Welcome bonus
* Description

Retrieval pipeline:

```text
Customer Query
      ↓
Sentence Transformer
      ↓
Embedding
      ↓
ChromaDB
      ↓
Metadata Filtering
      ↓
Relevant Card Facts
```

The system distinguishes between:

**Retrieval relevance** — how relevant the retrieved card information is to the query.

**Recommendation suitability** — how suitable the card is for the specific customer.

## 🔗 Hybrid Recommendation

The final recommendation combines:

```text
Eligibility Constraints
        +
ML Suitability Score
        +
Semantic Retrieval
        +
Structured Card Metadata
        ↓
Final Recommendation Ranking
```

This separates factual retrieval from personalized ranking.

## 🛡️ Financial AI Guardrails

The LLM receives only:

1. Customer information
2. ML recommendation results
3. Retrieved verified card facts

The system validates generated explanations against structured card information.

The LLM must not invent:

* Annual fees
* Cashback rates
* Eligibility requirements
* Benefits
* Welcome bonuses
* Credit-score requirements
* Approval probabilities

Unsupported information is rejected or marked as unverifiable.

The system does **not** provide financial advice or guarantee card approval.

## 📈 Explainability

The recommendation engine provides explanations using:

* Feature importance
* Permutation importance
* SHAP where computationally practical

Recommendations distinguish between:

**Model Signals**

* Spending alignment
* Reward preference
* Fee affordability

**Business Rules**

* Income eligibility
* Credit-score eligibility

**Retrieved Facts**

* Card benefits
* Fees
* Restrictions

## 🧪 Evaluation & Error Analysis

The project evaluates:

* Recommendation ranking quality
* Retrieval relevance
* LLM groundedness
* Unsupported claims
* Recommendation errors

Errors are categorized into:

* Reward preference mismatch
* Fee mismatch
* Weak spending alignment
* Borderline eligibility
* Ranking error
* Poor retrieval
* Insufficient card information

## 🏗️ Production Architecture

A production implementation could be structured as:

```text
                    Customer / Application
                            │
                            ▼
                    ┌───────────────┐
                    │ FastAPI       │
                    │ Gateway       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Validation    │
                    └───────┬───────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Recommendation      │
                 │ Service             │
                 └──────────┬──────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        Candidate Generation      Feature Engine
                │                       │
                └───────────┬───────────┘
                            ▼
                    ML Ranking Model
                            │
                            ▼
                    Hybrid Retrieval
                            │
                            ▼
                    Grounded LLM
                            │
                            ▼
                     Guardrails
                            │
                            ▼
                 Personalized Top-K
```

Potential production technologies:

* PostgreSQL — transactional/customer data
* S3 — data and model artifacts
* Redis — caching
* PySpark — large-scale interaction processing
* Docker — containerization
* FastAPI — model serving
* MLflow — experiment/model tracking
* AWS — cloud deployment
* CI/CD — automated testing and deployment
* Monitoring — model and data-quality monitoring

These components represent the proposed production architecture; the notebook focuses on the components that are practical to execute locally/within Colab.

## 📁 Project Structure

```text
personalized-credit-card-recommender/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── customers.csv
│   ├── credit_cards.csv
│   └── customer_card_interactions.csv
│
├── notebooks/
│   └── credit_card_recommendation.ipynb
│
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── candidate_generation.py
│   ├── ranking_model.py
│   ├── recommender.py
│   ├── retriever.py
│   ├── llm_explainer.py
│   ├── guardrails.py
│   └── evaluation.py
│
├── api/
│   └── main.py
│
├── tests/
│   ├── test_features.py
│   ├── test_eligibility.py
│   ├── test_ranking.py
│   ├── test_retrieval.py
│   └── test_guardrails.py
│
├── models/
│   └── .gitkeep
│
└── docs/
    ├── architecture.png
    └── interview_notes.md
```

## 🛠️ Technology Stack

**Programming:** Python, SQL

**Data & ML:** Pandas, NumPy, Scikit-learn, PySpark

**Recommendation:** Candidate Generation, Ranking, Top-K Evaluation

**GenAI / RAG:** ChromaDB, Sentence Transformers, LLM APIs

**Explainability:** SHAP, Feature Importance

**Backend:** FastAPI, REST APIs

**Infrastructure:** Docker, AWS

**MLOps:** MLflow, Testing, Monitoring, CI/CD concepts

## ▶️ Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```text
notebooks/credit_card_recommendation.ipynb
```

The notebook covers the complete workflow from synthetic data generation to the final recommendation demo.

## 💡 Example Output

```text
CUSTOMER PROFILE
----------------
Income: ₹85,000/month
Credit Score: 780
Travel Spending: High
Dining Spending: Medium
Preferred Reward: Travel
Fee Preference: Low

ELIGIBILITY ANALYSIS
--------------------
Eligible Cards: 8
Filtered Cards: 4

FINAL RECOMMENDATIONS
---------------------

1. Travel Rewards Card
   Eligibility: Eligible
   ML Suitability Score: 0.87

   Reasons:
   + Strong travel spending alignment
   + High reward preference match
   + Good fee affordability

   Trade-off:
   - Annual fee may reduce value at lower spending levels

2. Cashback Card
   Eligibility: Eligible
   ML Suitability Score: 0.81

3. Dining Rewards Card
   Eligibility: Eligible
   ML Suitability Score: 0.76

GUARDRAIL STATUS
----------------
Verified against retrieved card facts.

DISCLAIMER
----------
This is an experimental recommendation system using synthetic data.
It is not financial advice and does not guarantee approval.
```

## ⚠️ Limitations

This project is an **educational engineering prototype**.

The customer-card interaction data is synthetically generated because real banking recommendation and acceptance data is not publicly available.

Therefore:

* Model performance cannot be interpreted as real banking performance.
* Synthetic preferences may not represent real customer behavior.
* Real deployment would require validated historical interaction data.
* Financial-domain regulatory and compliance requirements would need dedicated review.
* Production systems would require privacy, security, monitoring, and human oversight.

## 🎯 Future Improvements

* Train on real anonymized customer-card interactions
* Implement a true learning-to-rank model
* Add online recommendation feedback
* Implement A/B testing
* Add model drift monitoring
* Add real-time feature serving
* Scale interaction processing with Spark
* Deploy recommendation API using AWS
* Add automated MLflow experiment tracking

## 📌 Disclaimer

This project is intended for **educational and technical demonstration purposes only**.

It uses synthetic experimental data and should not be interpreted as a banking recommendation engine, financial advice system, credit approval system, or representation of real-world banking performance.
