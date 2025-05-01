import numpy as np
import nltk as nlp
import random
from collections import defaultdict

# Download necessary NLTK resources once
nlp.download('punkt')
nlp.download('averaged_perceptron_tagger')

class SubjectiveTest:

    def __init__(self, data, noOfQues):  # Fixed constructor
        self.question_pattern = [
            "Explain in detail ",
            "Define ",
            "Write a short note on ",
            "What do you mean by "
        ]
        self.grammar = r"""
            CHUNK: {<NN.*|JJ>*<NN.*>}  # Improved noun chunk detection
        """
        self.summary = data
        self.noOfQues = int(noOfQues)

    @staticmethod
    def word_tokenizer(sequence):
        return [w for sent in nlp.sent_tokenize(sequence) for w in nlp.word_tokenize(sent)]

    def generate_test(self):
        sentences = nlp.sent_tokenize(self.summary)
        chunk_parser = nlp.RegexpParser(self.grammar)
        question_answer_dict = defaultdict(str)

        for sentence in sentences:
            words = nlp.word_tokenize(sentence)
            if len(words) < 6:  # Skip very short sentences
                continue

            tagged = nlp.pos_tag(words)
            tree = chunk_parser.parse(tagged)

            for subtree in tree.subtrees():
                if subtree.label() == "CHUNK":
                    keyphrase = " ".join(w for w, t in subtree).strip()
                    if len(keyphrase.split()) >= 2:  # Avoid trivial keywords
                        question_answer_dict[keyphrase] += sentence + " "

        if not question_answer_dict:
            return [], []

        # Prepare questions from selected keyphrases
        all_keys = list(question_answer_dict.keys())
        sample_size = min(self.noOfQues, len(all_keys))
        selected_keys = random.sample(all_keys, sample_size)

        questions = []
        answers = []

        for key in selected_keys:
            question = f"{random.choice(self.question_pattern)}{key}."
            answer = question_answer_dict[key].strip()
            questions.append(question)
            answers.append(answer)

        return questions, answers
