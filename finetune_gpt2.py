import os
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Define function to load a text dataset for language modeling
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
        overwrite_cache=True
    )

# Paths to your training and validation text files
train_file = "/data/ALU_train.txt"  # Contains your ALU info & conversational pairs
val_file = "/data/ALU_val.txt"      

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

# If you want to use your custom tokenizer from token_code.py, you could load it like this:
# from transformers import PreTrainedTokenizerFast
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

# Create datasets
train_dataset = load_dataset(train_file, tokenizer, block_size=128)
val_dataset = load_dataset(val_file, tokenizer, block_size=128) if os.path.exists(val_file) else None

# Create a data collator for language modeling (no masking for GPT-2)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training hyperparameters using TrainingArguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-alu",
    overwrite_output_dir=True,
    num_train_epochs=3,                  # Adjust number of epochs as needed
    per_device_train_batch_size=2,       # Adjust batch size based on your hardware
    per_device_eval_batch_size=2,
    evaluation_strategy="steps" if val_dataset else "no",
    eval_steps=500,                      # Evaluation frequency (if using validation)
    logging_steps=100,
    save_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",                    # Set to "tensorboard" or "wandb" if desired
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start fine-tuning the model
trainer.train()

# Save the fine-tuned model and tokenizer for later use
model.save_pretrained("./gpt2-finetuned-alu")
tokenizer.save_pretrained("./gpt2-finetuned-alu")
