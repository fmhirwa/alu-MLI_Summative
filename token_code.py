# Import necessary classes from tokenizers and transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Step 1: Initialize the tokenizer with the BPE model and specify the unknown token
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Step 2: Create a trainer with special tokens you want to include
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Step 3: Set the pre-tokenizer to split text by whitespace
tokenizer.pre_tokenizer = Whitespace()

# Step 4: Define the list of file paths that the tokenizer will be trained on.
# Replace the file paths with your actual data files (make sure the file exists)
files = ["data/file1.txt"]

# Step 5: Train the tokenizer on the specified files using the trainer
tokenizer.train(files, trainer)

# Optionally, save the trained tokenizer to a JSON file for future re-use
tokenizer.save("tokenizer.json")

# Step 6: Load the trained tokenizer into the Transformers library as a fast tokenizer
# Option A: Directly from the tokenizer object
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

# Option B: Alternatively, load from the saved JSON file (uncomment the next line if needed)
# fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

# Example usage: Encode some sample text using the fast tokenizer
sample_text = "Welcome to African Leadership University."
encoded = fast_tokenizer.encode(sample_text)

# Print the encoded token IDs directly (encoded is a list of token IDs)
print("Encoded IDs:", encoded)

# Alternatively, you can use encode_plus for more detailed output
encoded_plus = fast_tokenizer.encode_plus(
    sample_text,
    add_special_tokens=True,
)
print("Encoded IDs using encode_plus:", encoded_plus["input_ids"])

# Print the tokenized output (tokens as strings)
print("Tokenized Output:", fast_tokenizer.tokenize(sample_text))
