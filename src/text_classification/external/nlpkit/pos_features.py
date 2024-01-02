import collections

from sklearn.base import BaseEstimator, TransformerMixin


class POSExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts Parts of Speech (POS) counts for a collection of text documents.
    """

    def __init__(self, tagger):
        """
        :param tagger: a StanfordPosTagger object
        """
        self.tagger = tagger

    def transform(self, X, y=None):
        """
        :param X: a collection of docs
        :param y:
        :return: a list of dictionaries (pos_tag -> raw counts) corresponding to documents
        """
        return self._count_pos(X, self.tagger)

    def fit(self, X, y=None):
        return self

    def _count_pos(self, docs, tagger):
        pos = []
        for doc in docs:
            # Pos tag doc tokens
            tags = tagger.tag(doc.split())

            # A Counter object to store the frequencies of pos tags in a doc
            pos_counter = collections.Counter()
            for tag in tags:
                pos_counter[tag[1]] += 1

            pos.append(pos_counter)
        return pos


class POSTagPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pre-processes text documents by tagging each word in the form of word_TAG_
    e.g. what_WP. Can be used to generate POS tagged n-grams.
    """

    def __init__(self, tagger):
        """
        :param tagger: a StanfordPosTagger object
        """
        self.tagger = tagger

    def transform(self, X, y=None):
        """
        :param X: a collection of docs
        :param y:
        :return: a list of tagged docs
        """
        return self._pos_tag_docs(X, self.tagger)

    def fit(self, X, y=None):
        return self

    def _pos_tag_docs(self, docs, tagger):
        processed_docs = []
        for doc in docs:
            # Pos tag doc tokens
            tags = tagger.tag(doc.split())

            tagged_doc = []
            # tag each word in the form word_TAG_ e.g. what_WP
            for word_tag in tags:
                tagged_word = word_tag[0] + '_' + word_tag[1]
                tagged_doc.append(tagged_word)

            tagged_doc = ' '.join(tagged_doc)
            processed_docs.append(tagged_doc)

        return processed_docs
