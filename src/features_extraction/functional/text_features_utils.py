from __future__ import annotations

import html.entities
import json
import logging
import re

import spacy
from spacy import Language
from treetaggerwrapper import TreeTagger

from src.utils.yaml_manager import load_yaml

punctuation = r"""!"'\(\)\*\+,-\./;<=>\?\[\\]^_`{|}\~"""  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""


def char_to_unicode(char):
    return "U+" + format(ord(char), "04X")


def separate_html_entities(text) -> str:
    # Regular expression pattern for HTML entities and emojis
    pattern = r"(&#x[\da-fA-F]+;|&#\d+;)"

    # Add spaces around matches
    modified_text = re.sub(pattern, r" \1 ", text)

    # Remove potential extra spaces
    clean_text = " ".join(modified_text.split())

    return clean_text


def replace_with_unicode(text, mapping: dict):
    # Regular expression pattern for HTML entities and emojis
    pattern = r'(&#x[\da-fA-F]+;|&#\d+;)'

    # Find all matches
    matches = re.findall(pattern, text)

    for match in matches:
        # Unescape HTML entity and convert to Unicode
        unicode = char_to_unicode(html.unescape(match))
        try:
            code = mapping[unicode]
        except KeyError:
            print(f"Can't find {unicode}")
            pass
        else:
            # Replace match with Unicode in text
            text = text.replace(match, code)

    return text


class TextFeatureExtractor:
    def __init__(self, language: str = "it"):
        experiment_config: dict = load_yaml("src/features_extraction/config.yml")

        global TREETAGGER
        if experiment_config["tree_tagger_path"]:
            logging.warning(" ********** TreeTagger option is selected. "
                            " ********** This requires installing 'treetaggerwrapper' and TreeTagger.\n"
                            " ********** See https://pypi.org/project/treetaggerwrapper for instructions. ")
            TREETAGGER = TreeTagger(TAGDIR=experiment_config["tree_tagger_path"], TAGLANG=language)
        else:
            TREETAGGER = None

        self.__spacy_model = spacy.load(f"{language}_core_{'news' if language == 'it' else 'web'}_lg")
        if TREETAGGER is not None:
            self.__spacy_model.replace_pipe("lemmatizer", "tree_tagger")
        with open("dataset/asset/full-emoji-list.json", mode="r", encoding="utf-8") as f:
            emap = json.load(f)
        emap = [a for v in emap.values() for a in v]

        match = re.compile(r"[^\w]")
        self.__entity_map = {e["code"]: f":{match.sub('', e['description']).upper().strip()}:" for e in emap}

    def preprocessing_tokenizer(self, string: str) -> list[str]:

        # Separate EMOJIS from adjacent words if necessary
        string_clean = separate_html_entities(string)
        # Replace EMOJIS and ENTITIES with codes like ":CODE:"
        string_clean = replace_with_unicode(string_clean, self.__entity_map)
        # Remove all substrings with < "anything but spaces" >
        string_clean = re.sub("<\S+>", "", string_clean, flags=re.RegexFlag.IGNORECASE).strip()
        # Replace punctuation with space
        string_clean = re.sub(punctuation, " ", string_clean).strip()
        # Remove double spaces
        string_clean = re.sub(" +", " ", string_clean)

        # Regular expression pattern with negative lookahead (remove all characters that are not A-z, 0-9,
        # and all strings made of ":A-Z:", removing the colons
        string_clean = re.sub(r"(?!:[A-Z]+:)[^\w\s]|_", "", string_clean)  # removed !? for now
        string_clean = re.sub(r":", "", string_clean).strip()

        # Remove @card@
        # string_clean = string_clean.replace("@card@", "")

        string_empty = len(string_clean) == 0
        if string_empty:
            # Encode empty string with this string
            string_clean = "EMPTYSTRING"

        doc = self.__spacy_model(string_clean)
        tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop]
        tokens = [t for t in tokens if t and (not t.isalnum() or len(t) > 2) and t != "@card@"]

        return tokens


@Language.component("tree_tagger")
def tree_tagger(doc):
    """ Tagger component of TreeTagger compatible with Spacy pipeline """
    tokens = [token.text for token in doc if not token.is_space]

    tags = TREETAGGER.tag_text(tokens, tagonly=True)
    lemmas = [tag.split("\t")[2].split("|")[0] for tag in tags]

    j = 0
    for token in doc:
        if not token.is_space:
            token.lemma_ = lemmas[j]
            j += 1
        else:
            token.lemma_ = " "

    return doc
