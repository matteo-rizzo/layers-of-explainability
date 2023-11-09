from sklearn.base import BaseEstimator, TransformerMixin


class LIWCExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts proportions of words that fall in the various LIWC categories for a collection of text documents
    """

    def __init__(self, liwc, category_filter=None):
        """
        :param liwc: a LIWC object
        :param category_filter: a set of LIWC categories of interest to act as filter.
                                If a filter is not specified, word proportions for all
                                categories will be extracted.
        """
        self.liwc = liwc
        self.category_filter = category_filter

    def transform(self, X, y=None):
        """
        :param X: a list of docs
        :param y:
        :return: A list of dictionaries (category -> word proportions) corresponding to each doc
        """
        return self.__count_liwc_categories(X, self.liwc, self.category_filter)

    def fit(self, X, y=None):
        return self

    def __count_liwc_categories(self, docs, liwc, category_filter):
        docs_liwc_cats = []  # Holds a dict with category -> word proportions for each doc

        for doc in docs:
            tokens = doc.split()

            # All categories found in the document with raw word counts
            cats = liwc.parse(tokens)

            # Calculate proportions in each category
            for cat, count in cats.items():
                cats[cat] = count / len(tokens)

            if category_filter is not None:
                # Substract filtered categories from all categories, and remove
                # unwanted categories from the dict
                all_cats = set(liwc.categories.values())
                unwanted_cats = all_cats.difference(category_filter)
                for unwanted_cat in unwanted_cats:
                    # Use None to catch errors if key not exists
                    cats.pop(unwanted_cat, None)

            docs_liwc_cats.append(cats)

        return docs_liwc_cats
