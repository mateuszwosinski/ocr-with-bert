import os
import time
import string
import re
from typing import Dict, Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import easyocr

from modules.visualize import plot_bboxes
from modules.corrector import TypoCorrector_simple, TypoCorrector_contextual, TypoCorrector_BERT


pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # config line

class OCRSingleImage():
    def __init__(self,
                 lang: str = 'eng',
                 ocr_method: str = 'tesseract',
                 correction_method: str = None):
        assert ocr_method in ['tesseract', 'easy'], 'Selected OCR method not implemented!'
        assert correction_method in [None, 'simple', 'contextual', 'bert'], 'Selected Typo correction method not implemented!'
        
        self.lang = lang
        self.ocr_method = ocr_method
        self.correction_method = correction_method 
        
        if ocr_method == 'easy':
            if lang == 'eng':
                self.ocr = easyocr.Reader(['en'])

        if correction_method == 'simple':
            self.corrector = TypoCorrector_simple()
        elif correction_method == 'contextual':
            self.corrector = TypoCorrector_contextual()
        elif correction_method == 'bert':
            self.corrector = TypoCorrector_BERT(topk=200)
        elif correction_method is None:
            self.corrector = lambda x: x.split(' ')
        else:
            raise NotImplementedError
        
    def ocr_image(self,
                  img_path: str,
                  lang: str = 'eng',
                  plot: bool = False,
                  plot_save: bool = False
                  ) -> str:
        
        ocr_img = cv2.imread(img_path)
        
        if self.ocr_method == 'tesseract':
            ocr_data = pytesseract.image_to_data(ocr_img, lang=lang, output_type=Output.DICT)
        elif self.ocr_method == 'easy':
            ocr_data = self.ocr.readtext(img_path)
            
        ocr_data = self._convert_ocr_data(ocr_data)
        if len(ocr_data['conf']) == 0:
            print('!! Did not find any words on the image !!')
            return None
        
        ocr_text = ' '.join(ocr_data['text'])
        ocr_text = self.corrector(ocr_text)
        ocr_data['text'] = ocr_text
        
        if plot:
            split_path = os.path.splitext(img_path)
            out_path = f"{split_path[0]}_{self.correction_method}{split_path[1]}"
            ocr_img = plot_bboxes(ocr_img, ocr_data, out_path, save=plot_save)
        
        #ocr_text = ' '.join(ocr_data['text']).lower().replace('  ', ' ')
        return ocr_text
    
    @staticmethod
    def _convert_ocr_data(ocr_data: Dict[str, Any]
                          ) -> Dict[str, Any]:
        idxs_to_convert = []
        for idx, word in enumerate(ocr_data['text']):
            if word != '':
                idxs_to_convert.append(idx)
        
        keys_to_convert = ['left', 'top', 'width', 'height', 'text', 'conf']
        ocr_d = {}
        for k in keys_to_convert:
            ocr_d[k] = np.array(ocr_data[k])[idxs_to_convert].tolist()
        return ocr_d
    
def preprocess_text(text: str):
    pass
