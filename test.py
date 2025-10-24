from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-api-key-here",  # Replace with your OpenRouter API key
)

# Sample input data
user_cart = ["Apple iPhone 15", "JBL Headphones"]
user_views = ["Samsung Galaxy S23", "boAt Earbuds"]
recommendations = ["Spigen iPhone 15 Case", "JBL Bluetooth Speaker"]

# Prompt for the LLM
prompt = (
    f"The user's cart contains: {', '.join(user_cart)}. "
    f"They have also viewed: {', '.join(user_views)}. "
    f"The recommended products are: {', '.join(recommendations)}. "
    "Please explain the reasons for recommending each product, referencing the cart and views."
)

completion = client.chat.completions.create(
    extra_headers={
        # Optional, for your site's ranking on openrouter.ai
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>",
    },
    extra_body={},
    model="deepseek/deepseek-chat-v3.1:free",  # DeepSeek v3.1 free model
    messages=[{"role": "user", "content": prompt}],
)

print(completion.choices[0].message.content)
