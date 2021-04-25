import string
from typing import Dict

import numpy as np
import pandas as pd

from modules.corrector import TypoCorrector_simple, TypoCorrector_contextual, TypoCorrector_BERT


class TypoEvaluator():
    
    def __init__(self,
                 correction_method: str = 'bert'):
        if correction_method == 'bert':
            self.corrector = TypoCorrector_BERT(topk=2000)
        elif correction_method == 'simple':
            self.corrector = TypoCorrector_simple()
        elif correction_method == 'contextual':
            self.corrector = TypoCorrector_contextual()
        elif correction_method == 'none':
            self.corrector = lambda x: x.split(' ')

    def evaluate_single_text(self,
                             ocr_text: str,
                             true_text: str,
                             ) -> Dict[str, float]: 
        corrected_text = ' '.join(self.corrector(ocr_text))
        jaccard_sim = self.jaccard_similarity(corrected_text.translate(str.maketrans("","", string.punctuation)),
                                              true_text.translate(str.maketrans("","", string.punctuation)))
        return {'corrected_text': corrected_text,
                'jaccard': jaccard_sim}
    
    def evaluate_text_file(self,
                           eval_path: str,
                           ) -> Dict[str, float]:

        df_eval = pd.read_csv(eval_path)
        df_eval = df_eval.iloc[:100]
        similarities = {'jaccard': []}
        for _, row in df_eval.iterrows():
            text_similiarities = self.evaluate_single_text(row['OCR'],
                                                           row['True'])
            similarities['jaccard'].append(text_similiarities['jaccard'])
        
        out_similarities = {k: np.mean(v) for k, v in similarities.items()}
        return out_similarities
    
    def evaluate_random_text(self,
                             eval_path: str,
                             ) -> Dict[str, float]:
        df = pd.read_csv(eval_path)
        df_random = df.sample(n=1)
        ocr_text = df_random.iloc[0][1]
        true_text = df_random.iloc[0][0]
        
        out_dict = {'ocr_text': ocr_text,
                    'true_text': true_text}
        out_dict.update(self.evaluate_single_text(ocr_text, true_text))
        return out_dict
    
    def evaluate_text_from_string(self,
                                  text: str
                                  ):
        corrected_text = ' '.join(self.corrector(text))
        return corrected_text

    @staticmethod    
    def jaccard_similarity(query: str,
                           document: str
                           ) -> float:
        query = set(query.split(' '))
        document = set(document.split(' '))
        intersection = query.intersection(document)
        return round(float(len(intersection)) / (len(query) + len(document) - len(intersection)), 4)
        
         
def prepare_data(file_path: str,
                 eval_path: str):
    df = pd.read_csv(file_path)
    df = df[['Clean_Text', 'Corrupted_Text']]
    df['Same_len'] = df.apply(lambda x: len(x[0].split(' ')) == len(x[1].split(' ')), axis=1)
    df = df[df['Same_len'] == True].reset_index(drop=True)
    df = df.drop('Same_len', axis=1)
    df = df.rename({'Clean_Text': 'True',
                    'Corrupted_Text': 'OCR'}, axis='columns')
    df.to_csv(eval_path, index=False)
    return df
