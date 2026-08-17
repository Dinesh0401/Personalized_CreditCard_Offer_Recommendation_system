
def validate_output(explanation):
    # Simple check for hallucination keywords or specific flags
    hallucination_triggers = ['guaranteed', 'zero interest forever']
    for trigger in hallucination_triggers:
        if trigger in explanation.lower():
            return False
    return True
