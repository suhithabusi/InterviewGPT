from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found")
    exit()

print("✅ API key found")

client = Groq(
    api_key=api_key
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "What is Python? Explain in 2 simple sentences."
        }
    ],
    temperature=0.2,
    max_tokens=200
)

print("\n🤖 LLaMA Response:\n")
print(response.choices[0].message.content)