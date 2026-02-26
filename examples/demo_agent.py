"""
Demo Agent — A simple customer support chatbot for testing OpenEval.

This is a fake LLM agent that we use to test our evaluation framework.
It simulates a real chatbot with intentional flaws so scorers have 
something meaningful to evaluate.

Usage:
    from examples.demo_agent import chatbot, DATASET

    result = Eval(
        name="chatbot-eval",
        data=DATASET,
        task=chatbot,
        scorers=[ExactMatchScorer(), LLMJudgeScorer(...)],
    )
"""

# ─── The "Agent" (simple rule-based chatbot) ──────────────────────

KNOWLEDGE_BASE = {
    "return policy": "We offer a 30-day full refund policy at no extra cost.",
    "shipping": "Free shipping on orders over $50. Standard delivery takes 3-5 business days.",
    "hours": "We are open Monday through Friday, 9am to 5pm EST.",
    "payment": "We accept Visa, Mastercard, American Express, and PayPal.",
    "cancel subscription": "Go to Settings > Subscription > Cancel. Your access continues until the billing period ends.",
    "pro plan": "The Pro plan costs $49 per month and includes unlimited projects and priority support.",
    "free trial": "Yes, we offer a 14-day free trial with full access to all features.",
    "contact": "You can reach us at support@example.com or call 1-800-555-0123.",
}


def chatbot(input: str) -> str:
    """Simple keyword-matching chatbot. Intentionally imperfect."""
    input_lower = input.lower()

    for keyword, answer in KNOWLEDGE_BASE.items():
        if keyword in input_lower:
            return answer

    # Fallback: sometimes hallucinate (intentional!)
    if "phone support" in input_lower:
        # HALLUCINATION: Pro plan doesn't actually include phone support
        return "Yes, the Pro plan includes 24/7 phone support and a personal account manager."

    if "discount" in input_lower:
        # HALLUCINATION: making up a discount that doesn't exist
        return "We currently have a 40% discount on all annual plans!"

    return "I'm not sure about that. Let me connect you with a human agent."


# ─── Test Dataset ─────────────────────────────────────────────────

DATASET = [
    {
        "input": "What is your return policy?",
        "expected_output": "We offer a 30-day full refund policy at no extra cost.",
        "tags": ["policy", "easy"],
    },
    {
        "input": "How much does shipping cost?",
        "expected_output": "Free shipping on orders over $50. Standard delivery takes 3-5 business days.",
        "tags": ["shipping", "easy"],
    },
    {
        "input": "What are your business hours?",
        "expected_output": "We are open Monday through Friday, 9am to 5pm EST.",
        "tags": ["hours", "easy"],
    },
    {
        "input": "How do I cancel my subscription?",
        "expected_output": "Go to Settings > Subscription > Cancel. Your access continues until the billing period ends.",
        "tags": ["account", "medium"],
    },
    {
        "input": "How much does the Pro plan cost?",
        "expected_output": "The Pro plan costs $49 per month and includes unlimited projects and priority support.",
        "tags": ["pricing", "easy"],
    },
    {
        "input": "Does the Pro plan include phone support?",
        "expected_output": "The Pro plan includes priority support but phone support is not mentioned.",
        "tags": ["pricing", "hard", "hallucination-trap"],
    },
    {
        "input": "Do you have any discounts right now?",
        "expected_output": "I don't have information about current discounts.",
        "tags": ["pricing", "hard", "hallucination-trap"],
    },
    {
        "input": "What is quantum computing?",
        "expected_output": "I'm not sure about that, this is outside my knowledge area.",
        "tags": ["out-of-scope", "medium"],
    },
]

# ─── Context for RAG evaluation ──────────────────────────────────

RAG_CONTEXT = list(KNOWLEDGE_BASE.values())
"""Pass this as context to test Faithfulness scorer."""
