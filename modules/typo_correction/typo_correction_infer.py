import string
from typing import List

import numpy as np
import torch
import nltk
from transformers import BertForMaskedLM, BertTokenizer


class TypoCorrector():
    
    def __init__(self,
                 topk: int = 100):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.topk = topk
        
    def __call__(self,
                 masked_text: str,
                 org_words: List[str]
                 ):
        tokenized_text = self.tokenizer.tokenize(masked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        mask_ids = [ix for ix, word in enumerate(tokenized_text) if word == '[MASK]']
        
        #segments_ids = self._find_segments(masked_text.split(' '), tokenized_text)
        segments_ids = [0] * len(tokenized_text)

        segments_tensors = torch.tensor([segments_ids])
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        
        corrected_text = self.predict_masked_words(masked_text,
                                                   predictions,
                                                   mask_ids,
                                                   org_words)
        return corrected_text
            
    def predict_masked_words(self,
                             org_text: str,
                             predictions,
                             mask_ids: List[int],
                             org_words: List[str]
                             ):
        for mask, org_word in zip(mask_ids, org_words):
            preds = torch.topk(predictions[0][0][mask], k=self.topk)
            indices = preds.indices.tolist()
            predicted_words = self.tokenizer.convert_ids_to_tokens(indices)
            predicted_words = [s.translate(str.maketrans('','',string.punctuation)) for s in predicted_words]
            predicted_words = [s for s in predicted_words if s]
            best_word = predicted_words[np.argmin([nltk.edit_distance(org_word, pred_word) for pred_word in predicted_words])]
            org_text = org_text.replace('[MASK]', best_word, 1)
        return org_text            

    @staticmethod
    def _find_segments(org_text: List[str],
                       tokenized_text: List[str]
                       ) -> List[int]:
        """Split tokenized text into sentences based on first upper letter"""
        ix = 0
        segment_value = -1
        segments = []
        for token in tokenized_text:
            if token.startswith('##'):
                segments.append(segment_value)
                continue
            if (org_text[ix][0].isupper()) or (ix == 0):
                segment_value += 1
            segments.append(segment_value)
            ix += 1 
        return segments