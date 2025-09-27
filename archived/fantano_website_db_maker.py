import os
import warnings

from openai import OpenAI
from dotenv import load_dotenv

def extract_fantano_rating(html: str, max_length: int = 1000) -> str:

    if len(html) > max_length:
        warnings.warn(f"HTML is too long, truncating to final {max_length} characters. Results may suffer as a result.")
        html = html[-max_length:]

    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instructions_file_path = os.path.join(script_dir, "instructions", "fantano_website_db_maker.txt")

    with open(instructions_file_path, "r", encoding="utf-8") as f:
        instructions = f.read()

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=html,
        max_output_tokens=250,
        temperature=0.01,
    )

    return response.output_text