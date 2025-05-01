import re
import nltk
import numpy as np
from nltk.corpus import wordnet as wn, stopwords
from functools import lru_cache

# Ensure required NLTK resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

class ObjectiveTest:

    def __init__(self, data, noOfQues):
        self.summary = data
        self.noOfQues = noOfQues
        self.grammar = r"""
            CHUNK: {<NN.*>+<IN|DT>*<NN.*>+}
                {<NN.*>+<IN|DT>*<NNP.*>+}
                {<NNP.*>+<NNS>*}
        """
        self.chunker = nltk.RegexpParser(self.grammar)

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        trivial_sentences = []
        for sent in sentences:
            trivial = self.identify_trivial_sentences(sent)
            if trivial:
                trivial_sentences.append(trivial)
        return trivial_sentences

    def identify_trivial_sentences(self, sentence):
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)

        if tags[0][1] == "RB" or len(tokens) < 4:
            return None

        tree = self.chunker.parse(tags)
        noun_phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                phrase = " ".join(word for word, _ in subtree).strip()
                noun_phrases.append(phrase)

        replace_nouns = []
        for phrase in noun_phrases:
            phrase_tokens = nltk.word_tokenize(phrase)
            clean_tokens = [w for w in phrase_tokens if w.lower() not in stop_words and w.isalpha()]
            if clean_tokens:
                replace_nouns = clean_tokens
                break

        if not replace_nouns:
            return None

        val = min(len(i) for i in replace_nouns)

        trivial = {
            "Answer": " ".join(replace_nouns),
            "Key": val,
            "Similar": self.answer_options(replace_nouns[0]) if len(replace_nouns) == 1 else []
        }

        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = " ".join(["__________"] * len(replace_nouns))
        expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
        sentence = expression.sub(blanks_phrase, sentence, count=1)
        trivial["Question"] = sentence

        return trivial

    @staticmethod
    @lru_cache(maxsize=512)
    def answer_options(word):
        synsets = wn.synsets(word, pos="n")
        if not synsets:
            return []

        synset = synsets[0]
        if not synset.hypernyms():
            return []

        hypernym = synset.hypernyms()[0]
        hyponyms = hypernym.hyponyms()

        similar_words = []
        for hyponym in hyponyms:
            similar_word = hyponym.lemmas()[0].name().replace("_", " ")
            if similar_word.lower() != word.lower():
                similar_words.append(similar_word)
            if len(similar_words) == 8:
                break
        return similar_words

    def generate_test(self):
        trivial_pair = self.get_trivial_sentences()
        question_answer = trivial_pair

        if not question_answer:
            return [], []

        if len(question_answer) < int(self.noOfQues):
            self.noOfQues = len(question_answer)

        np.random.shuffle(question_answer)
        selected = question_answer[:int(self.noOfQues)]

        question = [item["Question"] for item in selected]
        answer = [item["Answer"] for item in selected]

        return question, answer

