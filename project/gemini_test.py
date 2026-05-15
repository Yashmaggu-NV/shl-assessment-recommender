import os
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

def generate_with_retry(prompt, model="grok-3-mini-fast", retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Max retries reached. Try again later.")

# Test it
if __name__ == "__main__":
    result = generate_with_retry("Say hello")
    print(result)