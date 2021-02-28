import spacy
from scipy import spatial
nlp = spacy.load('en_core_web_md') 

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

x = nlp.vocab['king'].vector
y = nlp.vocab['man'].vector
z = nlp.vocab['woman'].vector

#vectors closest to king-man+woman
new = x - y + z
compSimilarities = []

for s in nlp.vocab.vectors:
    vocab=nlp.vocab[s]

for word in nlp.vocab:
    if word.has_vector:       #only words with vectors
        if word.is_lower:     #only lowercase characters
            if word.is_alpha:  
                similarity = cosine_similarity(new, word.vector)
                compSimilarities.append((word, similarity))

compSimilarities = sorted(compSimilarities, key=lambda item: -item[1])

#print only first 10 words
g = [w[0].text for w in compSimilarities[:10]]
print(g)
