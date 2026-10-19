#!/usr/bin/env python3
"""Run baseline with priority vectors added to prompts"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datetime import datetime
from copy import deepcopy

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

def add_priorities_to_prompt(base_prompt, priorities):
    """Add priority information to persona prompt"""
    priority_text = f"""

My priorities on key issues (0-10 scale):
- Economy: {priorities['economy']}/10
- Healthcare: {priorities['healthcare']}/10
- Immigration: {priorities['immigration']}/10
- Climate Change: {priorities['climate']}/10
- Education: {priorities['education']}/10"""

    return base_prompt + priority_text

def ask_question(tokenizer, model, persona_prompt, question_text):
    """Query model"""
    prompt = f"""{persona_prompt}

Survey Question: {question_text}

Your answer (0-100):"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
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

def run_baseline_with_priorities(personas_config, priorities_config, survey_config,
                                 num_runs=10, output_dir="data/results"):
    """Run baseline with priorities added"""

    tokenizer, model = load_model()
    results = []

    # Create priority lookup
    priority_lookup = {p["persona_id"]: p["priorities"] for p in priorities_config}

    for persona in personas_config["personas"]:
        persona_id = persona["id"]
        persona_name = persona["name"]

        print(f"\n[{persona_name}]")
        print("=" * 80)

        base_prompt = build_persona_prompt(persona)
        priorities = priority_lookup.get(persona_id, {})

        # Add priorities to prompt
        augmented_prompt = add_priorities_to_prompt(base_prompt, priorities)

        print(f"  Running {num_runs} iterations: ", end="", flush=True)
        for run in range(num_runs):
            print(f"{run+1}", end=" ", flush=True)

            for question in survey_config["questions"]:
                answer = ask_question(
                    tokenizer,
                    model,
                    augmented_prompt,
                    question["question"]
                )

                results.append({
                    "persona_id": persona_id,
                    "persona_name": persona_name,
                    "q_id": question["q_id"],
                    "response": answer,
                    "run": run,
                    "with_priorities": True,
                    "timestamp": datetime.now().isoformat()
                })

        print()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"baseline_responses_with_priorities_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[✓] Augmented baseline responses saved to: {output_file}")
    print(f"[✓] Total responses: {len(results)}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priorities_file", required=True)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--output_dir", default="data/results")
    args = parser.parse_args()

    try:
        personas = json.load(open("config/personas.json"))
        survey = json.load(open("config/survey_config.json"))
        with open(args.priorities_file) as f:
            priorities = json.load(f)
    except FileNotFoundError as e:
        print(f"[✗] Config file not found: {e}")
        exit(1)

    run_baseline_with_priorities(personas, priorities, survey,
                                num_runs=args.num_runs, output_dir=args.output_dir)
