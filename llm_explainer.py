
def generate_grounded_explanation(recommendation, customer):
    facts = recommendation['fact_sheet']
    score = recommendation['suitability_score']

    prompt = f"""
    [SYSTEM GUARDRAIL: ONLY USE PROVIDED FACTS. NO HALLUCINATION.]
    Recommended: {recommendation['card_name']}
    Score: {score:.2f}
    Facts: {facts}
    """
    return prompt
