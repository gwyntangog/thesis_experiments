#!/usr/bin/env python3
"""Elicit priorities from each persona"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    """Load Mistral"""
    print(f"[Loading Model] {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("[✓] Model loaded successfully")
    return tokenizer, model

def build_persona_prompt(persona_dict):
    """Build persona prompt from template"""
    template = persona_dict["template"]
    values = persona_dict["values"]
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace(f"[{key}]", str(value))
    return prompt

def elicit_priorities(tokenizer, model, persona_prompt):
    """Ask persona to rank priorities"""

    prompt = f"""{persona_prompt}

I'm going to ask you to rank political issues by how important they are to you.

For each issue, give a priority score from 0 (not important) to 10 (very important).

Issues:
1. Economy
2. Healthcare
3. Immigration
4. Climate Change
5. Education

Respond with just the 5 numbers separated by spaces (e.g., "8 7 5 3 9")"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_new_tokens=20,
                temperature=0.5,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_part = response[len(prompt):].strip()

        # Extract numbers
        numbers = [int(s) for s in answer_part.split() if s.isdigit()]

        # Should have 5 priorities
        if len(numbers) >= 5:
            priorities = {
                "economy": min(10, max(0, numbers[0])),
                "healthcare": min(10, max(0, numbers[1])),
                "immigration": min(10, max(0, numbers[2])),
                "climate": min(10, max(0, numbers[3])),
                "education": min(10, max(0, numbers[4])),
            }
        else:
            # Default if parsing fails
            priorities = {
                "economy": 5,
                "healthcare": 5,
                "immigration": 5,
                "climate": 5,
                "education": 5,
            }

        return priorities
    except Exception as e:
        print(f"[Error] {e}")
        return {
            "economy": 5,
            "healthcare": 5,
            "immigration": 5,
            "climate": 5,
            "education": 5,
        }

def run_priority_elicitation(personas_config, output_dir="data/results"):
    """Elicit priorities from all personas"""

    tokenizer, model = load_model()
    results = []

    for persona in personas_config["personas"]:
        persona_id = persona["id"]
        persona_name = persona["name"]

        print(f"\n[{persona_name}]")
        print("=" * 80)

        persona_prompt = build_persona_prompt(persona)

        print(f"  Eliciting priorities...", end="", flush=True)
        priorities = elicit_priorities(tokenizer, model, persona_prompt)
        print(" ✓")

        print(f"    Economy: {priorities['economy']}/10")
        print(f"    Healthcare: {priorities['healthcare']}/10")
        print(f"    Immigration: {priorities['immigration']}/10")
        print(f"    Climate: {priorities['climate']}/10")
        print(f"    Education: {priorities['education']}/10")

        results.append({
            "persona_id": persona_id,
            "persona_name": persona_name,
            "priorities": priorities,
            "timestamp": datetime.now().isoformat()
        })

    # Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"elicited_priorities_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[✓] Priorities saved to: {output_file}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    try:
        personas = json.load(open("config/personas.json"))
    except FileNotFoundError as e:
        print(f"[✗] Config file not found: {e}")
        exit(1)

    run_priority_elicitation(personas)
