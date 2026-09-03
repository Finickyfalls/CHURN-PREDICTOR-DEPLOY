# Wraps the retention email generator built and tested in Part 2. Reads the
# GROQ_API_KEY from the environment -- set as a Railway environment variable
# in the deployed app, never committed to the repo.

import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

AGENT_SYSTEM_PROMPT = """You are a customer retention specialist at a telecom company.
You write short, warm, non-desperate retention emails to customers flagged as at risk of
cancelling. Rules:
- Under 120 words.
- Reference the customer's specific situation using the risk factors provided -- never
  invent details you were not given.
- Offer exactly ONE concrete, specific incentive relevant to their situation (e.g. a
  discount tied to their contract type, or a free add-on relevant to a service they lack).
- Never sound desperate or use guilt. Tone: genuinely helpful, not salesy.
- If a customer name is provided, greet them by name. If no name is provided, use a
  warm generic greeting like "Hi there" -- never write a placeholder like "[Name]".
- Sign off as "The [Company] Customer Care Team".
- Output ONLY the email body -- no subject line, no explanation, no markdown."""


def generate_retention_email(customer_summary, churn_probability, top_drivers, company_name="GreenLine Telecom", customer_name=None):
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
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content
