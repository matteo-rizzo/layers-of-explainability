import logging

import numpy as np
import torch
from nltk.tokenize import TweetTokenizer

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)


class SHAPexplainer:

    def __init__(self, model, tokenizer, words_dict, words_dict_reverse, device="cuda", use_logits: bool = True,
                 max_tokens: int = 512):
        self.model = model.model
        self.tokenizer = tokenizer
        self.device = device
        self.tweet_tokenizer = TweetTokenizer()
        self.words_dict = words_dict
        self.words_dict_reverse = words_dict_reverse
        self.use_logits = use_logits
        self.model_max_tokens = max_tokens

    def predict(self, indexed_words):

        sentences = [[self.words_dict[xx] if xx != 0 else "" for xx in x] for x in indexed_words]
        indexed_tokens, _, _ = self.tknz_to_idx(sentences)
        tokens_tensor = torch.tensor(indexed_tokens).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=tokens_tensor)
            logits = outputs[0]
            predictions = logits.detach().cpu().numpy()
        if self.use_logits:
            # return raw logits for additive shapley
            return predictions
        else:
            # return probabilities
            final = [softmax(x) for x in predictions]
            return np.array(final)

    def split_string(self, string):
        data_raw = self.tweet_tokenizer.tokenize(string)
        data_raw = [x for x in data_raw if x not in ".,:;'"]
        return data_raw

    def tknz_to_idx(self, train_data, MAX_SEQ_LEN=None):
        tokenized_nopad = [self.tokenizer.tokenize(" ".join(text), max_length=self.model_max_tokens, truncation=True)
                           for text in train_data]
        if not MAX_SEQ_LEN:
            MAX_SEQ_LEN = min(max(len(x) for x in train_data), self.model_max_tokens)
        tokenized_text = [['[PAD]', ] * MAX_SEQ_LEN for _ in range(len(tokenized_nopad))]
        for i in range(len(tokenized_nopad)):
            tokenized_text[i][0:len(tokenized_nopad[i])] = tokenized_nopad[i][0:MAX_SEQ_LEN]
        indexed_tokens = np.array([np.array(self.tokenizer.convert_tokens_to_ids(tt)) for tt in tokenized_text])
        return indexed_tokens, tokenized_text, MAX_SEQ_LEN

    def dt_to_idx(self, data, max_seq_len=None, truncate: bool = False):
        idx_dt = [[self.words_dict_reverse[xx] for xx in x] for x in data]
        if not max_seq_len:
            max_seq_len = min(max(len(x) for x in idx_dt), self.model_max_tokens)
        for i, x in enumerate(idx_dt):
            if len(x) < max_seq_len:
                idx_dt[i] = x + [0] * (max_seq_len - len(x))
            elif truncate:
                idx_dt[i] = x[:max_seq_len]
        return np.array(idx_dt), max_seq_len
