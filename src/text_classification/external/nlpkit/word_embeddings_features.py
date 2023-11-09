import numpy as np

from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class WordEmbedsDocVectorizer(BaseEstimator, TransformerMixin):
    """
    Converts documents to word2vec based document vector representations.
    It maps the words of a document to word2vec vectors, and averages them
    across dimensions to produce a document vector representation.
    """

    def __init__(self, word2vec, tfidf_weights=False):
        """
        :param word2vec: a gensim word2vec model
        :param tfidf_weights: If set to True, each word will be weighted according to
                        its tf-idf score, when calculating the document vector
                        representation. By default this is set to False,
                        and all words are assumed to have an equal weight of 1.
        """
        self.word2vec = word2vec
        self.tfidf_weights = tfidf_weights

    def transform(self, X, y=None):
        """
        :param X: a collection of docs
        :param y:
        :return: a np array where each row corresponds to a document, and the number
                 of columns equals the w2v model's dimensions.
        """
        return self._extract_w2v_features(X, self.word2vec, self.tfidf_weights)

    def fit(self, X, y=None):
        return self

    def _extract_w2v_features(self, docs, word2vec, tfidf_weights=False):
        # Convert text to a word2vec vector
        rows = []
        if tfidf_weights:
            # Generate a tfidf matrix that will act as weight for the generated
            # features
            # analyzer set to nltk word tokenize as default ommits token under
            # two characters
            tv = TfidfVectorizer(analyzer=word_tokenize)
            tfidf_matrix = tv.fit_transform(docs).toarray()

            for i in range(0, len(docs)):
                doc_tfidf_scores = self._get_doc_tfidf_scores(
                    tfidf_matrix, i, tv.get_feature_names())
                vector = self._convert_string2vec(docs[i], doc_tfidf_scores)
                rows.append(vector)
        else:
            for i in range(0, len(docs)):
                vector = self._convert_string2vec(docs[i])
                rows.append(vector)

        return rows

    def _convert_string2vec(self, doc, doc_tfidf_scores=None):
        """
        This method maps each word of a string to a word2vec vector,
        and returns a new vector which is the average of all word vectors in the string.

        :param doc:
        :param doc_tfidf_scores:
        :return: a list with a length that equals the word2vec model's dimensions, and
                where each element represents a dimension of the document.
        """
        # Tokenize string and convert tokens to lowercase
        tokens = word_tokenize(doc)
        tokens = [token.lower() for token in tokens]

        vectors = []  # Holds the word2vec representation for words in a string
        weights = []  # Holds the tfidf weight for words in a string
        if doc_tfidf_scores:
            for token in tokens:
                # Check if the word exists in the loaded word2vec model
                if token in self.word2vec.vocab:
                    vec = self.word2vec.get_vector(token)
                    vectors.append(vec)

                    weight = doc_tfidf_scores.get(token)
                    weights.append(weight)
        else:
            for token in tokens:
                if token in self.word2vec.vocab:
                    vec = self.word2vec.get_vector(token)
                    vectors.append(vec)

                    # Assume equal weight of 1
                    weights.append(1)

        # Todo: Temp fix if no words were found in the word2vec model
        # vocabulaly
        if len(vectors) == 0:
            return list(0 for i in range(0, self.word2vec.vector_size))

        # Return the average of all found word vectors
        return list(np.average(a=vectors, axis=0, weights=weights))

    @staticmethod
    def _get_doc_tfidf_scores(tfidf_matrix, doc_i, vocab):
        """
        This method takes the document's position in the corpus, and the extracted
        vocabulary, and returns a dictionary with words and Tf-idf frequencies.
        This methods joins the feature, id table produced produced by the TfidfVectorizer,
        and the Tf-idf matrix, in which features are encoded with ids.

        :param tfidf_matrix: a tf-idf matrix produced by the TfidfVectorizer
        :param doc_i: the document's index in the corpus
        :param vocab: the extracted vocabulary from the tfidf vectorizer
        :return: a dictionary with words as keys and tf-idf frequencies as values (word -> tf-idf)
        """
        # Find the indexes of all words that exist in the document
        indexes = tfidf_matrix[doc_i, :].nonzero()[0]

        # Fetch all scores from these indexes
        scores = [tfidf_matrix[doc_i, i] for i in indexes]

        # Array of all extracted features (word names)
        names = [vocab[i] for i in indexes]

        return dict(zip(names, scores))

    def _get_column_names(self):
        """
        This method returns the list of dimensions in the format of d1, d2,...dn.
        This list can be used in a pandas dataframe to name the columns according to
        the word2vec dimensions eg.
                df = pd.DataFrame(rows, columns=columns)

        :return: a list with the dimensions of a w2v model in the format of d{n}
        """
        return list(map(lambda i: "d{0}".format(
            i), range(0, self.word2vec.vector_size)))
