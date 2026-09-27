import json
import os
import random
from datetime import datetime
import openai  # or anthropic, or huggingface
import argparse

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def build_persona_prompt(persona, traits):
    """Create a system prompt that describes the persona"""
    trait_str = ", ".join([f"{k}: {v}" for k, v in traits.items()])
    return f"""You are roleplaying as a person with these characteristics: {trait_str}.

Please answer the following survey questions as this person would. Be consistent with your stated characteristics.
Respond with just the numeric answer for each question."""

def ask_question(client, persona_prompt, question_text, q_id, run_number):
    """Query LLM for a single question"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or your model
            messages=[
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question_text}
            ],
            temperature=0.7,  # Some randomness for realism
            max_tokens=50
        )
        answer = response.choices[0].message.content.strip()
        return {
            "q_id": q_id,
            "answer": answer,
            "run": run_number,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def run_experiment(personas_config, survey_config, num_runs=10, output_dir="data/results"):
    """Run survey for each persona multiple times"""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = []

    for persona in personas_config["personas"]:
        persona_id = persona["id"]
        traits = persona["traits"]
        persona_prompt = build_persona_prompt(persona, traits)

        print(f"Testing {persona['name']}...")

        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")

            for question_group in survey_config.values():
                for question in question_group:
                    result = ask_question(
                        client,
                        persona_prompt,
                        question["question"],
                        question["q_id"],
                        run
                    )
                    if result:
                        result["persona_id"] = persona_id
                        result["persona_name"] = persona["name"]
                        results.append(result)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"survey_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=10)
    args = parser.parse_args()

    personas = load_config("config/personas.json")
    survey = load_config("config/survey_config.json")

    run_experiment(personas, survey, num_runs=args.num_runs)
