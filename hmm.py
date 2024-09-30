import nltk
from nltk.corpus import brown
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Download required NLTK data
nltk.download('brown')
nltk.download('universal_tagset')

class HMMTagger:
    def __init__(self):
        self.tag_counts = defaultdict(int)
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, tagged_sentences):
        for sentence in tagged_sentences:
            prev_tag = '<START>'
            for word, tag in sentence:
                self.tag_counts[tag] += 1
                self.transition_counts[prev_tag][tag] += 1
                self.emission_counts[tag][word] += 1
                self.vocab.add(word)
                prev_tag = tag

        # After training, calculate unknown token emission for each tag
        for tag in self.tag_counts:
            self.emission_counts[tag]['<UNK>'] = 1  # A small probability for unseen words (can be tuned)

    def viterbi(self, sentence):
        V = [{}]
        path = {}
        
        # Initialize base cases (t == 0)
        for tag in self.tag_counts:
            # Use the 'UNKNOWN' token if the word is not in the emission counts
            first_word_emission_prob = self.emission_counts[tag].get(sentence[0], self.emission_counts[tag].get('<UNK>', 0))
            V[0][tag] = self.transition_counts['<START>'].get(tag, 0) * first_word_emission_prob
            path[tag] = [tag]
        
        # Run Viterbi for t > 0
        for t in range(1, len(sentence)):
            V.append({})
            newpath = {}
            for tag in self.tag_counts:
                (prob, state) = max((V[t-1][prev_tag] * 
                                     self.transition_counts[prev_tag].get(tag, 0) * 
                                     self.emission_counts[tag].get(sentence[t], self.emission_counts[tag].get('<UNK>', 0)), prev_tag) 
                                    for prev_tag in self.tag_counts)
                V[t][tag] = prob
                newpath[tag] = path[state] + [tag]
            path = newpath
        
        # Find the highest probability in the final state
        (prob, state) = max((V[len(sentence) - 1][tag], tag) for tag in self.tag_counts)
        return path[state]

    def tag(self, sentence):
        # Replace unknown words with the '<UNK>' token
        processed_sentence = [word if word in self.vocab else '<UNK>' for word in sentence]        
        # Run Viterbi algorithm on the processed sentence
        return self.viterbi(processed_sentence)

def prepare_data():
    tagged_sents = brown.tagged_sents(tagset='universal')
    print(f"Number of tagged sentences is {len(tagged_sents)}")
    return [([(word.lower(), tag) for word, tag in sent]) for sent in tagged_sents]


data = prepare_data()

tagger = HMMTagger()
tagger.train(data)