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
                 method: str = None):
        assert method in [None, 'simple', 'contextual'], 'Method not implemented yet!'
        
        self.lang = lang
        self.method = method
        
        if method == 'simple':
            self.corrector = TypoCorrector_simple()
        elif method == 'contextual':
            self.corrector = TypoCorrector_contextual()
        else:
            self.corrector = lambda x: x.split(' ')
        
    def ocr_image(self,
                  img_path: str,
                  lang: str = 'eng',
                  method = None
                  ) -> str:
        
        ocr_img = cv2.imread(img_path)
        
        ocr_data = pytesseract.image_to_data(ocr_img, lang=lang, output_type=Output.DICT)
        ocr_data = self._convert_ocr_data(ocr_data)
        ocr_text = ' '.join(ocr_data['text']).lower()
        
        ocr_data['text'] = self.corrector(ocr_text)

        ocr_img = plot_bboxes(ocr_img, ocr_data)
        ocr_text = ' '.join(ocr_data['text']).lower()
        return ocr_text
    
    @staticmethod
    def _convert_ocr_data(ocr_data: Dict[str, Any]
                          ) -> Dict[str, Any]:
        idxs = []
        for idx, word in enumerate(ocr_data['text']):
            if word != '':
                idxs.append(idx)
        
        ocr_d = {}        
        ocr_d['left'] = np.array(ocr_data['left'])[idxs].tolist()
        ocr_d['top'] = np.array(ocr_data['top'])[idxs].tolist()
        ocr_d['width'] = np.array(ocr_data['width'])[idxs].tolist()
        ocr_d['height'] = np.array(ocr_data['height'])[idxs].tolist()
        ocr_d['text'] = np.array(ocr_data['text'])[idxs].tolist()
        ocr_d['conf'] = np.array(ocr_data['conf'])[idxs].tolist()
        return ocr_d

OCR1 = OCRSingleImage(method=None)
OCR2 = OCRSingleImage(method='simple')
OCR3 = OCRSingleImage(method='contextual')

ocr_text1 = OCR1.ocr_image('examples/1.png', lang='eng')
print(f'No correction: {ocr_text1}')

ocr_text2 = OCR2.ocr_image('examples/1.png', lang='eng')
print(f'Simple correction: {ocr_text2}')

ocr_text3 = OCR3.ocr_image('examples/1.png', lang='eng')
print(f'Contextual correction: {ocr_text3}')


