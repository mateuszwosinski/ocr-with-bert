import os
import string
import random
from typing import Dict, Any

import pandas as pd
import numpy as np
from pipe import OCRSingleImage


class ImageEvaluator():
    
    def __init__(self,
                 ocr_method: str,
                 correction_method: str):
        self.corrector = OCRSingleImage(ocr_method=ocr_method, correction_method=correction_method)

    def evaluate_single_img(self,
                            df_words: pd.DataFrame,
                            img_path: str
                            ) -> Dict[str, Any]:
        img = img_path.split('/')[-1]
        try:
            true_text = df_words[df_words['file'] == img]['text'].tolist()[0]  
        except IndexError:
            print('Did not found ground true text for indicated image')
            return None, None
        
        corrected_text = ' '.join(self.corrector.ocr_image(img_path, lang='eng', plot=False, plot_save=False))
        jaccard = self.jaccard_similarity(corrected_text.translate(str.maketrans("","", string.punctuation)), 
                                          true_text.translate(str.maketrans("","", string.punctuation)))
        
        out_dict = {}
        out_dict['corrected_text'] = corrected_text
        out_dict['true_text'] = true_text
        out_dict['jaccard'] = jaccard
        return out_dict

    def evaluate_img_folder(self,
                            words_file: str,
                            images_folder: str,
                            num_imgs: int = 1):
        df_words = pd.read_csv(words_file)
        images = os.listdir(images_folder)
        
        similarities = {'jaccard': []}
        for ix, img in enumerate(images):
            out_dict = self.evaluate_single_img(df_words, os.path.join(images_folder, img))
            similarities['jaccard'].append(out_dict['jaccard'])
            if ix == num_imgs:
                break
        
        out_similarities = {k: np.mean(v) for k, v in similarities.items()}
        return out_similarities
            
    def evaluate_random_img(self,
                            words_file: str,
                            images_folder: str):
        
        df_words = pd.read_csv(words_file)  
        img = random.choice(df_words['file'])
        out_dict = self.evaluate_single_img(df_words, os.path.join(images_folder, img))
        return out_dict
    
    @staticmethod    
    def jaccard_similarity(query: str,
                           document: str
                           ) -> float:
        query = set(query.split(' '))
        document = set(document.split(' '))
        intersection = query.intersection(document)
        return round(float(len(intersection)) / (len(query) + len(document) - len(intersection)), 4)
