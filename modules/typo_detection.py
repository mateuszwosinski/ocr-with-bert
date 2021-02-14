import glob
import os
import re
import random

import nltk

class DatasetExtractor():
    
    def __init__(self):

        self._sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def extract_dataset(self,
                        files):
        labels = []  # 1 - if typo, 0 - if correct
        ocr_words = []
        gs_words = []
        
        for file in files:
            file_labels, file_ocr_words, file_gs_words = self.extract_file(file)
            labels.extend(file_labels)
            ocr_words.extend(file_ocr_words)
            gs_words.extend(file_gs_words)
        
        return ocr_words, gs_words, labels
    
    def extract_file(self,
                     file_path):
        
        with open(file_path, 'r') as f:
            raw_text = f.readlines()
        
        file_labels = []    
        file_ocr_words = []
        file_gs_words = []
        
        # ommit first 14 characters which contain the structure definition
        aligned_ocr = raw_text[1][14:]
        aligned_gs = raw_text[2][14:]
        
        sentence_spans = self._sentence_tokenizer.span_tokenize(aligned_ocr)
    
        for sentence_start, sentence_end in sentence_spans:
            sentence_labels = []
            sentence_ocr_words = []
            sentence_gs_words = []
            
            ocr_sentence = aligned_ocr[sentence_start: sentence_end]
            gs_sentence = aligned_gs[sentence_start: sentence_end]
            
            common_space_ids = self._get_common_space_ids(ocr_sentence, gs_sentence)
               
            word_start = 0
            for space_id in common_space_ids:
                ocr_word = ocr_sentence[word_start: space_id]
                gs_word = gs_sentence[word_start: space_id]

                if len(ocr_word) == 0:
                    word_start += 1
                    continue
                
                label = int(ocr_word != gs_word)
                sentence_labels.append(label)
                sentence_ocr_words.append(ocr_word)
                sentence_gs_words.append(gs_word)
                
                word_start = space_id + 1
                
            file_labels.append(sentence_labels)
            file_ocr_words.append(sentence_ocr_words)
            file_gs_words.append(sentence_gs_words)
            
        return file_labels, file_ocr_words, file_gs_words
    
    @staticmethod
    def _get_common_space_ids(ocr_sentence,
                              gs_sentence
                              ):
        
        ocr_space_ids = [match.span()[0] for match in re.finditer(" ", ocr_sentence)]
        gs_space_ids = [match.span()[0] for match in re.finditer(" ", gs_sentence)]
        
        common_space_ids = sorted(list(set(ocr_space_ids) & set(gs_space_ids)))
        common_space_ids.append(len(ocr_sentence))
        return common_space_ids
    
    def show_example(self, ocr_words, gs_words, labels):
        # the missing characters are defined by “@” sign
        rand_idx = random.randint(0, len(labels))
        
        ocr = ocr_words[rand_idx]
        gs = gs_words[rand_idx]
        lbl = labels[rand_idx]
        
        print(f'Sentence number {rand_idx}\n')
        for o, g, l in zip(ocr, gs, lbl):
            print(f'{o} --- {g} --- {l}')

    
train_files = sorted(glob.glob(os.path.join("../../data/training_18M_without_Finnish/EN", "*", "*.txt")))

Dataset = DatasetExtractor()
ocr_words, gs_words, labels = Dataset.extract_dataset(train_files)
Dataset.show_example(ocr_words, gs_words, labels)
            
