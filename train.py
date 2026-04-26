from transformers import pipeline

# 🔥 USE YOUR TRAINED MODEL
classifier = pipeline("text-classification", model="./model")

print(classifier("A man is running </s></s> A man is sleeping"))