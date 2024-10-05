import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('SAPLING_PRIVATE_API_KEY')


def tone_analyser(text):
    response = requests.post(
        "https://api.sapling.ai/api/v1/tone",
        json={
            "key": API_KEY,
            "text": text
        }
    )

    response = response.json()
    overall_tone = response["overall"]
    values = [item[0] for item in overall_tone]
    dominant_tone = max(values)
    for tone in overall_tone:
        if tone[0] == dominant_tone:
            tone = tone[1]
            break

    return tone
