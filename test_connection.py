from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{
        "role": "user",
        "content": "Say: BA Agent is ready to go!"
    }]
)

print(message.content[0].text)
