from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
response = pipe("Je suis un nouveau papa. Comment puis-je aider ma femme ?", max_new_tokens=100)
print(response[0]["generated_text"])
