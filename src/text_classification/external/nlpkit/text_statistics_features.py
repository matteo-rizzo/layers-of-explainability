from sklearn.base import BaseEstimator, TransformerMixin
from textstat.textstat import textstat as ts


class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """
    Calculates various text statistics and readability scores for a collection of text documents.
    """

    def transform(self, X, y=None):
        """
        :param X: A collection of docs
        :param y:
        :return: a list of dictionaries with calculated scores for each doc
        """
        return self._calculate_scores(X)

    def fit(self, X, y=None):
        return self

    def _calculate_scores(self, docs):
        docs_scores = []

        for doc in docs:
            scores = {}
            scores['chars'] = ts.char_count(doc)
            scores['words'] = ts.lexicon_count(doc)
            scores['sents'] = ts.sentence_count(doc)
            # scores['syllables'] = ts.syllable_count(doc)
            scores['avg_sent_length'] = ts.avg_sentence_length(doc)
            scores['avg_syllables_per_word'] = ts.avg_syllables_per_word(doc)
            scores['avg_letters_per_word'] = ts.avg_letter_per_word(doc)
            scores['flesch'] = ts.flesch_reading_ease(doc)
            # scores['smog'] = ts.smog_index(doc)
            scores['coleman_liau'] = ts.coleman_liau_index(doc)
            scores['automated_readability'] = ts.automated_readability_index(
                doc)
            # scores['linsear'] = ts.linsear_write_formula(doc)
            scores['difficult_words'] = ts.difficult_words(doc)
            scores['dale_chall'] = ts.dale_chall_readability_score(doc)
            # scores['gunning_fog'] = ts.gunning_fog(doc)
            scores['lix'] = ts.lix(doc)
            docs_scores.append(scores)

        return docs_scores
