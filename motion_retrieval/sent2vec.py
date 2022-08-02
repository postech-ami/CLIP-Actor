import torch
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer

class Sent2Vec():
    # Same base then split into two separate modules
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


    def compute_distance(self, prompt, raw_label, idx):
        sentences = []
        prefix = 'a human '
        sentences.append(prompt)
        sentences.append(prefix + raw_label[idx[0]])
        sentences.append(prefix + raw_label[idx[1]])
        sentences.append(prefix + raw_label[idx[2]])
        sentences_vectors = self.model.encode(sentences)

        dist = []
        dist.append(spatial.distance.cosine(sentences_vectors[0], sentences_vectors[1]))
        dist.append(spatial.distance.cosine(sentences_vectors[0], sentences_vectors[2]))
        dist.append(spatial.distance.cosine(sentences_vectors[0], sentences_vectors[3]))
        label_idx = np.argmin(np.array(dist))

        return raw_label[idx[label_idx]]

    def only_sent2vec(self, prompt, babel_raw_label):
        print(prompt)
        prefix = 'a human '

        sentences = [prefix + label for label in babel_raw_label]
        sentences.append(prompt)

        self.vectorizer.bert(sentences)
        vectors_bert = self.vectorizer.vectors
        prompt_vectors = np.repeat(np.expand_dims(vectors_bert[-1], axis=0), 7231, axis=0)
        sim = torch.cosine_similarity(torch.tensor(prompt_vectors), torch.tensor(vectors_bert[:-1]))
        idx = torch.argmax(sim)
        print(babel_raw_label[idx])

        return babel_raw_label[idx]

