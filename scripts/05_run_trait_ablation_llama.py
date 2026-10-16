#!/usr/bin/env python3
"""
Trait Ablation Experiment with Mistral
Remove traits one-at-a-time and measure response changes
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datetime import datetime
from copy import deepcopy

def load_llama_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    """Load Mistral from HuggingFace"""
    print(f"[Loading Model] {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
        )
        print("[✓] Model loaded successfully")
        return tokenizer, model
    except Exception as e:
        print(f"[✗] Error loading model: {e}")
        raise

def build_persona_prompt(persona_dict):
    """Build persona prompt from template + values"""
    template = persona_dict["template"]
    values = persona_dict["values"]

    # Substitute values into template
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace(f"[{key}]", str(value))

    return prompt

def ask_question_llama(tokenizer, model, persona_prompt, question_text):
    """Query Mistral for a single question"""

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

        if numbers:
            answer = max(0, min(100, numbers[0]))
        else:
            answer = 50

        return answer

    except Exception as e:
        print(f"[Error] Query failed: {e}")
        return 50

def create_ablated_persona(persona_dict, trait_to_remove):
    """Remove one trait from persona by replacing with neutral value"""
    ablated = deepcopy(persona_dict)

    # Neutral replacement values for each trait
    neutral_values = {
        "PID": "someone",
        "EDUCATION": "with some education",
        "AGE": "middle-aged",
        "RACETH": "of unspecified race/ethnicity",
        "INCOME": "making a moderate income",
        "IDEO": "of moderate political ideology",
        "MARST": "of unspecified marital status",
        "GENDER": "",
        "REGIS": "registered",
        "INTEREST": "regularly"
    }

    # Replace the trait with neutral value in the values dict
    if trait_to_remove in ablated["values"]:
        ablated["values"][trait_to_remove] = neutral_values.get(trait_to_remove, "unspecified")

    return ablated

def run_ablation_experiment(personas_config, survey_config, ablation_config, num_runs=5, output_dir="data/results"):
    """Run trait ablation experiment"""

    tokenizer, model = load_llama_model()
    results = []

    for persona in personas_config["personas"]:
        persona_id = persona["id"]
        persona_name = persona["name"]

        print(f"\n[{persona_name}]")
        print("=" * 80)

        # Get baseline (full persona)
        baseline_prompt = build_persona_prompt(persona)
        baseline_responses = {}

        print(f"  Baseline (Full Persona): ", end="", flush=True)
        for run in range(num_runs):
            print(f"{run+1}", end=" ", flush=True)

            for question in survey_config["questions"]:
                answer = ask_question_llama(
                    tokenizer,
                    model,
                    baseline_prompt,
                    question["question"]
                )

                q_id = question["q_id"]
                if q_id not in baseline_responses:
                    baseline_responses[q_id] = []
                baseline_responses[q_id].append(answer)

        print()

        # Ablate each trait
        traits_to_ablate = ablation_config.get("traits_to_ablate", ["PID", "EDUCATION", "AGE", "RACETH"])

        for trait_to_remove in traits_to_ablate:
            # Create ablated persona
            ablated_persona = create_ablated_persona(persona, trait_to_remove)
            ablated_prompt = build_persona_prompt(ablated_persona)
            ablated_responses = {}

            print(f"  Ablate '{trait_to_remove}': ", end="", flush=True)
            for run in range(num_runs):
                print(f"{run+1}", end=" ", flush=True)

                for question in survey_config["questions"]:
                    answer = ask_question_llama(
                        tokenizer,
                        model,
                        ablated_prompt,
                        question["question"]
                    )

                    q_id = question["q_id"]
                    if q_id not in ablated_responses:
                        ablated_responses[q_id] = []
                    ablated_responses[q_id].append(answer)

            print()

            # Compute effect sizes for this ablation
            for question in survey_config["questions"]:
                q_id = question["q_id"]

                baseline_mean = sum(baseline_responses[q_id]) / len(baseline_responses[q_id])
                ablated_mean = sum(ablated_responses[q_id]) / len(ablated_responses[q_id])

                effect_size = abs(ablated_mean - baseline_mean)

                results.append({
                    "persona_id": persona_id,
                    "persona_name": persona_name,
                    "trait_removed": trait_to_remove,
                    "q_id": q_id,
                    "baseline_mean": float(baseline_mean),
                    "ablated_mean": float(ablated_mean),
                    "effect_size": float(effect_size),
                    "percent_change": float((effect_size / baseline_mean * 100) if baseline_mean != 0 else 0),
                    "baseline_responses": baseline_responses[q_id],
                    "ablated_responses": ablated_responses[q_id],
                    "timestamp": datetime.now().isoformat()
                })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"ablation_results_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[✓] Ablation results saved to: {output_file}")
    print(f"[✓] Total ablation tests: {len(results)}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trait ablation experiment with Mistral")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs per ablation")
    parser.add_argument("--output_dir", default="data/results", help="Output directory")
    args = parser.parse_args()

    try:
        personas = json.load(open("config/personas.json"))
        survey = json.load(open("config/survey_config.json"))
        ablation = json.load(open("config/ablation_config.json"))
    except FileNotFoundError as e:
        print(f"[✗] Config file not found: {e}")
        exit(1)

    run_ablation_experiment(personas, survey, ablation, num_runs=args.num_runs, output_dir=args.output_dir)
