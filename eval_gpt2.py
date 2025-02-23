import os
import re
import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load fine-tuned GPT-2 model and tokenizer
model_dir = "./gpt2-finetuned-alu"
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
model.eval()

# Read and parse evaluation file (ALU_eval.txt)
with open("ALU_eval.txt", "r", encoding="utf-8") as f:
    eval_data = f.read()

# Regex pattern to extract prompt and reference answer pairs
pattern = r"\[Prompt \d+\]\s*Q:\s*(.*?)\s*A:\s*(.*?)\s*(?=\[Prompt \d+\]|$)"
matches = re.findall(pattern, eval_data, re.DOTALL)

prompts = []
references = []
for match in matches:
    prompt = match[0].strip()
    reference = match[1].strip()
    prompts.append(prompt)
    references.append(reference)

print(f"Found {len(prompts)} evaluation prompts.\n")

# Define a simple function to compute token-level F1 score
def compute_f1(prediction, reference):
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Initialize lists for metric scores and generated responses
bleu_scores = []
f1_scores = []
generated_responses = []

# Evaluate each prompt
for prompt, ref in zip(prompts, references):
    # Encode prompt and generate response
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=150, 
            num_beams=5, 
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove the prompt from the generated text if it is repeated
    if generated_text.startswith(prompt):
        gen_answer = generated_text[len(prompt):].strip()
    else:
        gen_answer = generated_text.strip()
    generated_responses.append(gen_answer)
    
    # Compute BLEU score using nltk (with smoothing)
    smoothing_fn = SmoothingFunction().method1
    reference_tokens = ref.split()
    candidate_tokens = gen_answer.split()
    bleu = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_fn)
    bleu_scores.append(bleu)
    
    # Compute F1 score
    f1 = compute_f1(gen_answer, ref)
    f1_scores.append(f1)
    
    # Print individual results
    print(f"Prompt: {prompt}")
    print(f"Reference: {ref}")
    print(f"Generated: {gen_answer}")
    print(f"BLEU: {bleu:.4f}, F1: {f1:.4f}")
    print("-" * 50)

# Chart the BLEU and F1 scores per prompt using matplotlib
x = list(range(1, len(prompts) + 1))

plt.figure(figsize=(12, 5))

# BLEU score chart
plt.subplot(1, 2, 1)
plt.bar(x, bleu_scores, color='skyblue')
plt.xlabel("Prompt Number")
plt.ylabel("BLEU Score")
plt.title("BLEU Scores per Prompt")

# F1 score chart
plt.subplot(1, 2, 2)
plt.bar(x, f1_scores, color='lightgreen')
plt.xlabel("Prompt Number")
plt.ylabel("F1 Score")
plt.title("F1 Scores per Prompt")

plt.tight_layout()
plt.show()

# Print average metrics
avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
print(f"Average BLEU Score: {avg_bleu:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
