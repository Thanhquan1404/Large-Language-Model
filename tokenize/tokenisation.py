from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokens = tokenizer.tokenize("The cat sat.")
print(tokens)
# ['The', 'Ġcat', 'Ġsat', '.']