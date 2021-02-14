import re
from collections import Counter


class TypoCorrector():
    
    def __init__(self,
                 corpus_file: str = 'big.txt'):
        self.WORDS = Counter(self.get_all_words(open(corpus_file).read()))
        self.N = sum(self.WORDS.values())
    
    @staticmethod
    def get_all_words(text: str
              ):
        return re.findall(r'\w+', text.lower())
    
    def get_word_prob(self,
                  word: str
                  ) -> float:
        return self.WORDS[word] / self.N
    
    def get_known(self,
                  words):
        return set(w for w in words if w in self.WORDS)
    
    def candidates(self,
                   word):
        return (self.get_known([word]) or self.get_known(self.edits1(word)) 
                or self.get_known(self.edits2(word)) or [word])
    
    def edits1(self,
               word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self,
               word):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def correction(self,
                   word): 
        return max(self.candidates(word), key=self.get_word_prob)
    
    def __call__(self,
                 word):
        return self.correction(word)
    
    
Corrector = TypoCorrector('../big.txt')
print(Corrector('corecton'))

    


