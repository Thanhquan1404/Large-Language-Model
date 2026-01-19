# load the dataset
with open("./tinyshakespeare.txt", "r") as file:
  corpus = file.read()

print(f"Text corpus includes {len(corpus.split())} word.")

# to simulate multiple documents, we chunk up the corpus into 10 pieces
N = len(corpus) // 10
documents = [corpus[i:i+N] for i in range (0, len(corpus), N)]

documents = documents[:-1] #last document is residue
# now we have N documents from the corpus
# Text corpus includes 202651 words.

from sklearn.feature_extraction.text import TfidfVectorizer

vectornizer = TfidfVectorizer()
embeddings = vectornizer.fit_transform(documents)
words = vectornizer.get_feature_names_out()

print(f"Word count: {len(corpus)} e.g.: {words[:10]}")
print(f"Embedding shape: {embeddings.shape}")

