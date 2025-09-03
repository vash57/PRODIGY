from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned")

prompt = input("Enter your prompt: ")

result = generator(
    prompt,
    max_new_tokens=100,
    temperature=0.7,       # lower temperature = less randomness
    top_p=0.9,             # nucleus sampling
    repetition_penalty=1.2, # penalize repeated words
    do_sample=True,
    eos_token_id=50256      # GPT-2 EOS token
)

print("\nGenerated Text:\n")
print(result[0]['generated_text'])
