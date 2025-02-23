import re
import json

# Input file containing Q/A pairs
input_file = "./data/ALU_train3.txt"
# Output file that will contain only the answers
output_file = "ALU_train_answers.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = f.read()

# Regex pattern to capture the answer following "A:" until the next blank line or the end of file.
# This assumes that each Q/A pair is separated by at least one blank line.
pattern = r"Q:.*?A:\s*(.*?)\n\s*\n"
matches = re.findall(pattern, data, re.DOTALL)

# If the file does not strictly have a blank line separator at the end, you can try:
if not matches:
    pattern = r"Q:.*?A:\s*(.*)"
    matches = re.findall(pattern, data, re.DOTALL)

with open(output_file, "w", encoding="utf-8") as out_f:
    for answer in matches:
        # Create a JSON object with the answer field
        record = {"answer": answer.strip()}
        out_f.write(json.dumps(record) + "\n")

print(f"Extracted {len(matches)} answers to {output_file}")
