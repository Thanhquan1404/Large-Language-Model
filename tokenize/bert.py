from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Unbelievable results!")
print(tokens)
# ['un', '##bel', '##iev', '##able', 'results', '!']
