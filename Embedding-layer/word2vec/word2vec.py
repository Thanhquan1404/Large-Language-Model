# let's load pretrained embedding and see how they look
import gensim 
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
# access to download: https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz

print(f"The embedding size: {model.vector_size}")
print(f"The vocabulary size: {len(model)}")

# italy - rome + london = england
model.most_similar(positive=['london', 'italy'], negative=['rome'])

### OUTPUT ###
# [('england', 0.5743448734283447),
#  ('europe', 0.537047266960144),
#  ('liverpool', 0.5141493678092957),
#  ('chelsea', 0.5138063430786133),
#  ('barcelona', 0.5128480792045593)]

model.most_similar(positive=['woman', 'doctor'], negative=['man'])
### OUTPUT ###
# [('gynecologist', 0.7093892097473145),
#  ('nurse', 0.6477287411689758),
#  ('doctors', 0.6471460461616516),
#  ('physician', 0.6438996195793152),
#  ('pediatrician', 0.6249487996101379)]