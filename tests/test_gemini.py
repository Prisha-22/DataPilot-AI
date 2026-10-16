import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.llm.gemini_client import client


interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Reply with exactly: DataPilot Gemini connection successful."
)

print(interaction.output_text)