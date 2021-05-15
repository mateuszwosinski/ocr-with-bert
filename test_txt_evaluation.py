# %reload_ext autoreload
# %autoreload 2
import nltk
from evaluate_text import TypoEvaluator

eval_path = 'data/evaluation/amazon_imdb.csv'
correction_method = 'bert'  # 'simple', 'contextual', 'bert' or 'none'

evaluator = TypoEvaluator(correction_method=correction_method)


random_out = evaluator.evaluate_random_text(eval_path)  # test on a single text from csv file


for key, val in random_out.items():
    print(f'{key}: {val}\n')


org_text = 'hile preparing for battle I hvae always found that plans are useles, but planing is indispensable.'
text = evaluator.evaluate_text_from_string(org_text)
print(f'\nOriginal text: {org_text} \nCorrected text: {text}')


# test different correction methods on a csv file
correctors = ['none', 'bert', 'simple', 'contextual']
random_out = {}
all_out = {}
for corr_method in correctors:
    evaluator = TypoEvaluator(correction_method=corr_method)
    all_out[corr_method] = evaluator.evaluate_text_file(eval_path)
    random_out[corr_method] = evaluator.evaluate_random_text(eval_path)
    print(f'Jaccard similarity for {corr_method} corrector: {all_out[corr_method]}')
    del evaluator


