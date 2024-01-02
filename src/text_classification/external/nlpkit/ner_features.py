import collections

from sklearn.base import BaseEstimator, TransformerMixin


class NERPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pre-processes text documents by replacing named entities with generic tags e.g.
        - PERSON
        - ORGANIZATION
        - LOCATION
    """

    def __init__(self, tagger):
        """
        :param tagger: A StanfordParser object
        """
        self.tagger = tagger

    def transform(self, X, y=None):
        """
        :param X: a list of documents
        :param y:
        :return: a list of documents with the named entities replaced by generic tags
        """
        return self._replace_entities(X, self.tagger)

    def fit(self, X, y=None):
        # Fit method is only required by scikit pipeline
        return self

    def _replace_entities(self, docs, tagger):
        processed_docs = []
        for doc in docs:
            ner_results = tagger.tag(doc.split())

            prev_entity = None
            tokens = []
            for token, entity_type in ner_results:
                if entity_type != 'O':
                    # By default "John Doe" will be replaced with PERSON PERSON
                    # If the the previous entity type matches current type they
                    # should merge to eg. PERSON
                    if prev_entity != entity_type:
                        # enclose in underscore to differentiate from words
                        entity_tag = '_' + entity_type + '_'
                        tokens.append(entity_tag)
                else:
                    tokens.append(token)
                prev_entity = entity_type

            processed_docs.append(' '.join(tokens))

        return processed_docs


class NamedEntitiesCounter(BaseEstimator, TransformerMixin):
    """
    Extracts Named Entity counts per entity type (e.g. PERSON) for a collection of text documents.
    """

    def __init__(self, tagger):
        """
        :param tagger: a CoreNLPParser object with tagtype='ner'
        """
        self.tagger = tagger

    def transform(self, X, y=None):
        """
        :param X: a collection of docs
        :param y:
        :return: a list of dictionaries (entity_type -> raw counts) corresponding to documents
        """
        return self._count_named_entities(X, self.tagger)

    def fit(self, X, y=None):
        return self

    def _count_named_entities(self, docs, tagger):
        ne = []
        for doc in docs:
            ner_results = tagger.tag(doc.split())

            # A Counter object to store the frequencies of pos tags in a doc
            ne_counter = collections.Counter()
            prev_entity = None
            for _, entity_type in ner_results:
                if entity_type == 'O':
                    ne_counter[entity_type] += 1
                else:
                    # fix for duplicate counts in case of (John, Person),
                    # (Doe,PERSON)
                    if prev_entity != entity_type:
                        ne_counter[entity_type] += 1
                prev_entity = entity_type

            ne.append(ne_counter)
        return ne
