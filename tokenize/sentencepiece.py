import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('m.model')

tokens = sp.encode("The cat sat.", out_type=str)
print(tokens)
# ['▁The', '▁cat', '▁sat', '.']