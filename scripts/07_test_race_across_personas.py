#!/usr/bin/env python3
"""Test if race specification affects responses across all 9 personas"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from copy import deepcopy

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    """Load Mistral from HuggingFace"""
    print(f"[Loading Model] {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("[✓] Model loaded successfully")
    return tokenizer, model

def build_persona_prompt(persona_dict):
    """Build persona prompt from template + values"""
    template = persona_dict["template"]
    values = persona_dict["values"]

    prompt = template
    for key, value in values.items():
        prompt = prompt.replace(f"[{key}]", str(value))

    return prompt

def ask_question(tokenizer, model, persona_prompt, question_text):
    """Query model for a single question"""
    prompt = f"""{persona_prompt}

Survey Question: {question_text}

Your answer (0-100):"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"].to(model.device),
                max_new_tokens=10,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_part = response[len(prompt):].strip()
        numbers = [int(s) for s in answer_part.split() if s.isdigit()]

        return max(0, min(100, numbers[0])) if numbers else 50
    except Exception as e:
        print(f"[Error] {e}")
        return 50

def run_race_test(personas_config, survey_config, output_dir="data/results"):
    """Test each persona with different race specifications"""

    tokenizer, model = load_model()
    results = []

    races = [
        "non-Hispanic white",
        "non-Hispanic black",
        "Hispanic",
        "of unspecified race/ethnicity"
    ]

    for persona in personas_config["personas"]:
        persona_id = persona["id"]
        persona_name = persona["name"]

        print(f"\n[{persona_name}]")
        print("=" * 80)

        for race in races:
            # Create persona with this race
            race_persona = deepcopy(persona)
            race_persona["values"]["RACETH"] = race

            race_prompt = build_persona_prompt(race_persona)

            print(f"  Testing race: {race:30s}", end=" ", flush=True)

            for question in survey_config["questions"]:
                answer = ask_question(
                    tokenizer,
                    model,
                    race_prompt,
                    question["question"]
                )

                results.append({
                    "persona_id": persona_id,
                    "persona_name": persona_name,
                    "race_tested": race,
                    "q_id": question["q_id"],
                    "response": answer,
                    "timestamp": datetime.now().isoformat()
                })

            print(f"✓")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"race_sensitivity_test_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[✓] Results saved to: {output_file}")
    print(f"[✓] Total tests: {len(results)}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    try:
        personas = json.load(open("config/personas.json"))
        survey = json.load(open("config/survey_config.json"))
    except FileNotFoundError as e:
        print(f"[✗] Config file not found: {e}")
        exit(1)

    run_race_test(personas, survey)
