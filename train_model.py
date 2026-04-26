import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

print("🚀 Training started...")

# 🔷 STEP 1: Load dataset
df = pd.read_csv("dataset.csv")

# 🔷 STEP 2: Clean dataset
df = df.dropna()

df["sentence1"] = df["sentence1"].astype(str)
df["sentence2"] = df["sentence2"].astype(str)
df["gold_label"] = df["gold_label"].astype(str)

df = df[df["gold_label"].isin(["contradiction", "entailment", "neutral"])]

# 🔥 IMPORTANT: Use ONLY ONE sampling
df = df.sample(2000)

print("✅ Dataset ready")
print(df.head())

# 🔷 STEP 3: Convert labels
label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
df["label"] = df["gold_label"].map(label_map)

# 🔷 STEP 4: Convert dataset
dataset = Dataset.from_pandas(df)

# 🔷 STEP 5: Tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
        padding="max_length"
    )

dataset = dataset.map(tokenize, batched=True)

# 🔷 STEP 6: Split
dataset = dataset.train_test_split(test_size=0.1)

# 🔷 STEP 7: Model
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=3
)

# 🔷 STEP 8: Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="none"
)

# 🔷 STEP 9: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# 🔷 STEP 10: Train
trainer.train()

# 🔷 STEP 11: Evaluate
results = trainer.evaluate()
print("📊 Evaluation Results:", results)

# 🔷 STEP 12: Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("✅ Model trained and saved in /model folder")