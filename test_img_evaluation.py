import os
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
