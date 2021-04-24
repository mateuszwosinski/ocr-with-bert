from evaluate_text import TypoEvaluator

eval_path = 'data/evaluation/amazon_imdb.csv'

evaluator = TypoEvaluator()
random_out = evaluator.evaluate_random_text(eval_path)  # test on a single text from csv file

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
