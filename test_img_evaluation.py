# %reload_ext autoreload
# %autoreload 2
import os
import time
from evaluate_images import ImageEvaluator

folder_path = 'data/cvl-database-1-1'
words_file = 'data_testset.csv'
images_folder = 'testset/pages'

evaluator = ImageEvaluator(ocr_method='tesseract',
                           correction_method='bert')

random_out = evaluator.evaluate_random_img(os.path.join(folder_path, words_file),
                                           os.path.join(folder_path, images_folder),
                                          plot=True)

for key, val in random_out.items():
    print(f'{key}: {val}\n')

ocr_methods = ['tesseract', 'htr_line', 'htr_word']
correction_methods = ['bert']
examples = ['examples_lines/a02-012-00.png']

for correction_method in correction_methods:
    for ocr_ix, ocr_method in enumerate(ocr_methods):
        evaluator = ImageEvaluator(ocr_method=ocr_method,
                                   correction_method=correction_method)
        for example in examples:
            if ocr_ix == 0:
                evaluator.show_image(example)
            try:
                ocr_text = ' '.join(evaluator.evaluate_img_from_path(example))
                print(f'\n{correction_method} correction, {ocr_method} ocr:\n{ocr_text}')
            except TypeError:
                print(f'\n{correction_method} correction, {ocr_method} ocr:\nNothing found.')

evaluator.evaluate_img_folder(os.path.join(folder_path, words_file),
                              os.path.join(folder_path, images_folder))
