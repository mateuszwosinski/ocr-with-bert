import os
import pandas as pd
from pipe import OCRSingleImage


def evaluate_single_img(df_words: pd.DataFrame,
                        img_path: str,
                        ocr_model):
    img = img_path.split('/')[-1]
    try:
        true_text = df_words[df_words['file'] == img]['text'].tolist()[0]  
    except IndexError:
        print('Did not found ground true text for indicated image')
        return None, None
    
    ocr_text = ocr_model.ocr_image(img_path, lang='eng', plot=False, plot_save=False)
      
    return ocr_text, true_text


def evaluate_folder(words_file: str,
                    images_folder: str,
                    ocr_method: str = 'tesseract',
                    correction_method: str = 'bert',
                    num_imgs: int = 1):
    df_words = pd.read_csv(words_file)
    images = os.listdir(images_folder)
    
    OCR = OCRSingleImage(ocr_method=ocr_method, correction_method=correction_method)
    for ix, img in enumerate(images):
        evaluate_single_img(df_words, os.path.join(images_folder, img), OCR)
        if ix == num_imgs:
            break
        
folder_path = 'data/cvl-database-1-1'
words_file = 'data_testset.csv'
images_folder = 'testset/pages'

evaluate_folder(os.path.join(folder_path, words_file),
                os.path.join(folder_path, images_folder))
    