from typing import List

import numpy as np
import spacy

SPACY_MODEL = spacy.load("it_core_news_lg")


def check_word_polarity(word: str, doc_original: str) -> float:
    """
    Check whether a give word in the doc representation as negative (-) or positive (+) polarity
    @param word: the vocabulary word related to the word representation
    @param doc_original: the original input text in free form
    @return -1 if word HAS relationship with "non", else 1 if word HAS NO relationship with "non". 0 if word does not
    appear in doc_original
    """
    tokenized_doc_original = SPACY_MODEL(doc_original)
    for token in tokenized_doc_original:
        if token.text == word:
            token_relations = [child.text.lower() for child in token.children]
            print(token.text, token.dep_, token.head.text, token.head.pos_, token_relations)
            return -1 if "non" in token_relations else 1
    return 0


def inject_polarity(doc_representation: np.ndarray, doc_original: str, vocabulary: List[str]) -> np.ndarray:
    """
    Transforms the representation ndarray produced through TFIDF by adding negative sign to the words that are related
    to a negation (e.g., "Non sei stronza" -> [0.3, 0.5, 0.8] -> [[0.3, -0.5, -0.8]
    @param doc_representation: the TFIDF float values representation of the input doc
    @param doc_original: the original input doc in free text
    @return: the doc representation where terms with negative polarity were multiplied by -1
    """
    for i, word_representation in enumerate(doc_representation):
        polarity = check_word_polarity(word=vocabulary[i], doc_original=doc_original)
        print(word_representation, polarity)
        doc_representation[i] = str(float(word_representation) * polarity)
    return doc_representation


def main():
    vocabulary = ["sborro", "ciao bella", "mannagialpapa", "dio stregone", "ultrasauro", "scopo"]
    # doc_original = "Non sei sborro"
    doc_original = "non ti scopo neanche col cazzo di un altro"
    doc_representation = np.array([0.0, 0, 0, 0, 0, 0.1])
    print("Before\n", doc_representation)
    doc_representation = inject_polarity(doc_representation, doc_original=doc_original, vocabulary=vocabulary)
    print("After\n", doc_representation)


if __name__ == "__main__":
    main()
