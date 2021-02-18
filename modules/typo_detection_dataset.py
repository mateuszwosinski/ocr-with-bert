import pickle
from typing import List, Any

import torch
import numpy as np
from transformers import BertTokenizer


class TypoDataset():
    def __init__(self,
                 max_sequence_length: int = 100):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_sequence_length = max_sequence_length
    
    def _tokenize_and_preserve_labels(self,
                                     sentence: List[str], 
                                     labels: List[int]):
        """ Tokenize sentence into BERT subtokens."""
        tokenized_sentence = []
        tokenized_labels = []
    
        for word, label in zip(sentence, labels):
    
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
    
            tokenized_sentence.extend(tokenized_word)
            tokenized_labels.extend([label] * n_subwords)

        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        tokenized_labels = [0] + tokenized_labels + [0]
        return tokenized_sentence, tokenized_labels
    
    def _truncate_or_pad(self,
                        arr: List[Any]
                        ):
        """ Truncate or pad the `arr` according the maximum sequence length"""
        
        return arr[:self.max_sequence_length] + [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(arr))

    def prepare_dataset(self,
                        words: List[str], 
                        labels: List[int],
                        ):
        """Extract inputs, tags and masks tensors from the dataset"""
    
        tokenized_words, tokenized_labels = [self._tokenize_and_preserve_labels(sentence, label) 
                                             for sentence, label in zip(words, labels)]
    
        input_ids = np.array([self._truncate_or_pad(self.tokenizer.convert_tokens_to_ids(sentence),
                                                    self.max_sequence_length) 
                              for sentence in tokenized_words], dtype='long')
        
        subtoken_labels = np.array([self._truncate_or_pad(sentence_labels,
                                                          self.max_sequence_length) 
                                    for sentence_labels in tokenized_labels], dtype='long')
        
        attention_masks = [[int(i != 0) for i in ids] for ids in input_ids]
    
        inputs = torch.tensor(input_ids, dtype=torch.long)
        tags = torch.tensor(subtoken_labels, dtype=torch.long)
        masks = torch.tensor(attention_masks, dtype=torch.long)
        return inputs, tags, masks
