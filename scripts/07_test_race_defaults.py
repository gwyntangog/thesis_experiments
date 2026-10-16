#!/usr/bin/env python3
"""Test if model defaults to white when race is removed"""

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def ask_question(tokenizer, model, prompt, question):
    full_prompt = f"{prompt}\n\nSurvey Question: {question}\n\nYour answer (0-100):"

    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"].to(model.device),
            max_new_tokens=10,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,  # Deterministic for comparison
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_part = response[len(full_prompt):].strip()
    numbers = [int(s) for s in answer_part.split() if s.isdigit()]
    return max(0, min(100, numbers[0])) if numbers else 50

# Load model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True,
)

# Test template
base_template = "It is 2024. You are a 45 year-old, married, [RACE] male with a college degree making $70k per year, living in the United States. You are a moderate, registered Democrat who regularly pays attention to what's going on in government and politics."

races = ["non-Hispanic white", "non-Hispanic black", "Hispanic", "of unspecified race/ethnicity"]
questions = [
    "How do you feel toward the Democratic Party? (0-100)",
    "How do you feel toward the Republican Party? (0-100)"
]

print("Testing if race specification changes responses:\n")

for question in questions:
    print(f"Q: {question}")
    for race in races:
        prompt = base_template.replace("[RACE]", race)
        response = ask_question(tokenizer, model, prompt, question)
        print(f"  {race:30s} → {response:3d}")
    print()
