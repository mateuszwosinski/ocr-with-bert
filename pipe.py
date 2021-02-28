import time
from typing import Dict, Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from modules.visualize import plot_bboxes
from modules.typo_correction import TypoCorrector_simple, TypoCorrector_contextual

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # config line


class OCRSingleImage():
    def __init__(self,
                 lang: str = 'eng',
                 detection_method: str = None,
                 correction_method: str = None):
        assert correction_method in [None, 'simple', 'contextual'], 'Method not implemented yet!'
        
        self.lang = lang
        self.detection_method = detection_method
        self.correction_method = correction_method
        
        if detection_method is None:
            self.detector = lambda x: x
        
        if correction_method == 'simple':
            self.corrector = TypoCorrector_simple()
        elif correction_method == 'contextual':
            self.corrector = TypoCorrector_contextual()
        elif correction_method is None:
            self.corrector = lambda x: x.split(' ')
        
    def ocr_image(self,
                  img_path: str,
                  lang: str = 'eng',
                  method = None,
                  plot: bool = False
                  ) -> str:
        
        ocr_img = cv2.imread(img_path)
        
        ocr_data = pytesseract.image_to_data(ocr_img, lang=lang, output_type=Output.DICT)
        ocr_data = self._convert_ocr_data(ocr_data)
        
        ocr_text = self.detector(ocr_data['text'])
        ocr_text = ' '.join(ocr_text).lower()
        
        ocr_data['text'] = self.corrector(ocr_text)
        
        if plot:
            ocr_img = plot_bboxes(ocr_img, ocr_data)
        
        ocr_text = ' '.join(ocr_data['text']).lower()
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

OCR1 = OCRSingleImage(correction_method=None)
OCR2 = OCRSingleImage(correction_method='simple')
OCR3 = OCRSingleImage(correction_method='contextual')

start1 = time.time()
ocr_text1 = OCR1.ocr_image('examples/1.png', lang='eng', plot=True)
print(f'\nNo correction:\n {ocr_text1}\nTime spent: {time.time() - start1}')

start2 = time.time()
ocr_text2 = OCR2.ocr_image('examples/1.png', lang='eng', plot=True)
print(f'\nSimple correction:\n {ocr_text2}\nTime spent: {time.time() - start2}')

start3 = time.time()
ocr_text3 = OCR3.ocr_image('examples/1.png', lang='eng', plot=True)
print(f'\nContextual correction:\n {ocr_text3}\nTime spent: {time.time() - start3}')


