import re
import os
from collections import Counter
from typing import List

import spacy
import contextualSpellCheck as SpellCheck


class TypoCorrector_simple():
    
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
                   word: str
                   ) -> str: 
        return max(self.candidates(word), key=self.get_word_prob)
    
    def __call__(self,
                 sentence: str
                 ) -> List[str]:
        return [self.correction(word) for word in sentence.split(' ')]
    
    
class TypoCorrector_contextual():
    
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.corrector = spacy.load('en_core_web_sm')
        SpellCheck.add_to_pipe(self.corrector)
        
    def __call__(self,
                 sentence: str
                 ) -> List[str]:
        doc = self.corrector(sentence)
        corrected_sentence = doc._.outcome_spellCheck
        return corrected_sentence.split(' ') 
    
# =============================================================================
# Corrector = TypoCorrector_simple('../big.txt')
# print(Corrector('corecton'))
# =============================================================================
    


