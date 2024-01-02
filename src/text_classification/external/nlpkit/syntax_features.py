from nltk import Tree
from sklearn.base import BaseEstimator, TransformerMixin


class CFGExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts the Context Free Grammar (CFG) production rules found in a collection of text documents.
    """

    def __init__(self, parser, include_lexical=True):
        """
        :param parser: An StanfordParser object
        :param include_lexical: Option to include or exclude production rules that contain
                                terminal tokens (words)
        """
        self.parser = parser
        self.include_lexical = include_lexical

    def transform(self, X, y=None):
        """
        :param X: a collection of docs
        :param y:
        :return: a list of lists that contain the production rules found in each
                 document
        """
        return self._extract_cfg_rules(X, self.include_lexical)

    def fit(self, X, y=None):
        return self

    def _extract_cfg_rules(self, docs, include_lexical, print_progress=False):
        productions = []
        i = 0
        for doc in docs:
            # Parse doc and generate a syntax tree
            parse_tree = Tree("START", self.parser.raw_parse(doc))
            # Extract the productionvrules that correspond to the non-terminal
            # nodes of the tree
            prods = parse_tree.productions()

            # Create a list of all productions in document
            doc_prods = []
            for prod in prods:
                if (str(prod) == "START -> ROOT"):  # skip this
                    continue
                if (include_lexical):
                    doc_prods.append(str(prod))
                else:
                    # note that is_lexical is not the opposite of is_nonlexical
                    if (prod.is_nonlexical()):
                        doc_prods.append(str(prod))

            productions.append(doc_prods)

            # track progress
            if (print_progress):
                if (i % 500) == 0:
                    print("{0}/{1} docs processed...".format(i, len(docs)))
            i += 1

        return productions


def dummy_tokenizer(doc):
    """
    A pass through method for pre-tokenized documents. Use it as preprocessor,
    and tokenizer for the tfidf/count vectorizer. (to replace lambda x:x)
    :param doc:
    :return:
    """
    return doc
