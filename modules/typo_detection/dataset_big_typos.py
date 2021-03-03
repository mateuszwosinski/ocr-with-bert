import re
import string


def prepare_data(text_path: str):
    big_file = open(text_path).read()

    regex_match = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
    sentences = regex_match.findall(big_file)
    sentences = [s.split('\n\n') for s in sentences]
    sentences = [s for sublist in sentences for s in sublist]
    sentences = [s.replace('\n', ' ') for s in sentences]
    sentences = [s.strip() for s in sentences]
    sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    sentences = [s.split(" ") for s in sentences]
    sentences = [s for s in sentences if len(s) > 2]

    return sentences

s = prepare_data('../../data/big.txt')
