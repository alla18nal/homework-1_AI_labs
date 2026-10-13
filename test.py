import os
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])
r = client.chat.completions.with_raw_response.create(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    messages=[{"role": "user", "content": "Say hi"}],
)
print(r.http_response.text)