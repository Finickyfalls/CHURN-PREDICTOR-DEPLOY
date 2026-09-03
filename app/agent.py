import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Two distinct agent "personalities" -- same hard rules (length, grounding in
# real risk factors, one concrete incentive, no desperation), different tone.
PERSONAS = {
    "warm": {
        "name": "Warm & Personal",
        "system_prompt": """You are a customer retention specialist at a telecom company with a warm,
personal communication style. You write short, warm, non-desperate retention emails to customers
flagged as at risk of cancelling. Rules:
- Under 120 words.
- Reference the customer's specific situation using the risk factors provided -- never invent
  details you were not given.
- Offer exactly ONE concrete, specific incentive relevant to their situation.
- Tone: genuinely warm and personal, like a helpful human who values the relationship -- use
  phrases that acknowledge their time as a customer, never salesy or corporate-sounding.
- If a customer name is provided, greet them by name. If no name is provided, use a warm generic
  greeting like "Hi there" -- never write a placeholder like "[Name]".
- Sign off as "The [Company] Customer Care Team".
- Output ONLY the email body -- no subject line, no explanation, no markdown."""
    },
    "professional": {
        "name": "Professional & Efficient",
        "system_prompt": """You are a customer retention specialist at a telecom company with a
crisp, professional communication style. You write short, professional, non-desperate retention
emails to customers flagged as at risk of cancelling. Rules:
- Under 120 words.
- Reference the customer's specific situation using the risk factors provided -- never invent
  details you were not given.
- Offer exactly ONE concrete, specific incentive relevant to their situation, stated clearly and
  early in the email.
- Tone: efficient, confident, and business-like -- minimal small talk, get to the point, respectful
  but not overly familiar.
- If a customer name is provided, greet them by name. If no name is provided, use a neutral
  greeting like "Hello" -- never write a placeholder like "[Name]".
- Sign off as "The [Company] Customer Care Team".
- Output ONLY the email body -- no subject line, no explanation, no markdown."""
    }
}


def _call_llm(system_prompt, customer_summary, churn_probability, top_drivers, company_name, customer_name):
    name_line = f"Customer name: {customer_name}" if customer_name else "Customer name: not provided"

    user_prompt = f"""Company name: {company_name}
{name_line}
Customer summary: {customer_summary}
Predicted churn probability: {churn_probability:.0%}
Top risk factors driving this prediction: {", ".join(top_drivers)}

Write the retention email now."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def generate_retention_email(customer_summary, churn_probability, top_drivers, company_name="GreenLine Telecom", customer_name=None, persona="warm"):
    """Single-persona generation -- kept for backward compatibility."""
    system_prompt = PERSONAS[persona]["system_prompt"]
    return _call_llm(system_prompt, customer_summary, churn_probability, top_drivers, company_name, customer_name)


def generate_email_variants(customer_summary, churn_probability, top_drivers, company_name="GreenLine Telecom", customer_name=None):
    """A/B generation: returns both persona variants for the same customer,
    so they can be compared side by side in the UI."""
    results = {}
    for key, persona in PERSONAS.items():
        email_text = _call_llm(
            persona["system_prompt"], customer_summary, churn_probability,
            top_drivers, company_name, customer_name
        )
        results[key] = {"persona_name": persona["name"], "email": email_text}
    return results
