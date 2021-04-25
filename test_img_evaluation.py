import os
import time

from evaluate_images import ImageEvaluator


folder_path = 'data/cvl-database-1-1'
words_file = 'data_testset.csv'
images_folder = 'testset/pages'

evaluator = ImageEvaluator(ocr_method='tesseract',
                           correction_method='bert')

random_out = evaluator.evaluate_random_img(os.path.join(folder_path, words_file),
                                           os.path.join(folder_path, images_folder))

#evaluator.evaluate_img_folder(os.path.join(folder_path, words_file),
#                              os.path.join(folder_path, images_folder))


ocr = 'tesseract'
correction_methods = [None, 'simple', 'contextual', 'bert']
examples = ['examples/1.png', 'examples/2.jpg', 'examples/3.jpg']

for correction_method in correction_methods:
    evaluator = ImageEvaluator(ocr_method=ocr,
                               correction_method=correction_method)
    for example in examples:
        start = time.time()
        ocr_text = evaluator.evaluate_img_from_path(example)
        print(f'\n{correction_method} correction:\n {ocr_text}\nTime spent: {time.time() - start}')
