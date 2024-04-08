import logging 

import spacy
import pyphen # for hyphenation

import numpy as np

from transformers import pipeline

from scipy.special import softmax
from scipy.stats import entropy


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

""" Class that manipulates only on the gap level.
    For a gap and word, this computes the respective feature values.
    Has an additional function returning the possible feature space 
    given a word for all possible gaps.
"""
class GapManipulator:
    def __init__(self, dictionary_path: str, update_bert: bool=False) -> None:
        self.nlp_spacy = spacy.load("en_core_web_sm")

        self.syllable_processor = pyphen.Pyphen(lang="en")
        self.dictionary = self.read_dictionary(dictionary_path)
        
        self.nlp = pipeline("fill-mask", model="bert-base-cased")
        self.nlp.top_k = 50
        self.update_bert = update_bert # Flag if we want to update the bert score or not.
        
        self.feature_index_dict = {
                                   "bert_gap_proba": 49,
                                   "bert_top50_entropy": 50,
                                   "compound_break": 56,
                                   "th_gap": 57,
                                   "syllable_break": 58,
                                   "length_of_solution_chars": 59,
                                   "length_of_solution_sylls": 60
                                   }
        
    def read_dictionary(self, dictionary_path: str) -> set:
        dictionary = set()
        with open(dictionary_path,'r') as lines:
            for line in lines:
                dictionary.add(line.strip().lower())
        
        return dictionary
                
    def get_num_syllables(self, word: str) -> int:
        # Get the number of syllables in a word
        if word.isalpha():
            return len(self.syllable_processor.inserted(word.lower()).split("-"))
            
        return 1 # if not alphanumeric, return 1.

    def is_syllable_break(self, word: str, gap: str) -> bool:
        # Checks if the gap leads to a syllable break
        if word.isalpha():
            syllables = self.syllable_processor.inserted(word.lower()).split("-")
            if len(syllables) <= 1:
                return 0.0
            checkstring = ""
            #Reverse list of syllables and check if they match with the gap.
            for syllable in reversed(syllables):
                checkstring = syllable + checkstring
                if checkstring == gap:
                    return 1.0 
            
        return 0.0

    def is_compound_break(self, word: str, gap: str) -> bool:
        # Check if we find the first and second half in the dictionary (then it is a compound break) 
        doc = self.nlp_spacy(word)
        if doc[0].pos_ not in ["PROPN","NOUN"]:
            return 0.0
        
        first_half = word[:-len(gap)]

        if first_half.lower() in self.dictionary and gap in self.dictionary and len(first_half) > 2 and len(gap) > 2:
            return 1.0
            
        return 0.0
        
    def get_bert_entropy_top50(self, word: str, gap: str) -> float:
        instance = word[:-len(gap)] + "{}".format(self.nlp.tokenizer.mask_token)
        result = self.nlp(instance)
        top50 = [x['score'] for x in result]
        ent = entropy(softmax(np.array(top50)))
        
        return ent
        
    def get_bert_token_proba(self, word, gap):
        instance = word[:-len(gap)] + "{}".format(self.nlp.tokenizer.mask_token)
        result = self.nlp(instance, targets=[gap])
        
        return result[0]['score']
        
    def get_th_gap(self, hint):
        # Returns 1 for th hint, else 0
        if hint.lower() == "th":
            return 1.0
        return 0.0
        
    def get_gap(self, word: str, gap_size: int) -> (str, str, list):
        # Set a new gap for a given word with a given size.
        # Returns the word, gap, and all related features.
        # NOTE: The gap size is capped to leave at least 1 character!
        word = word
        gap = word[-gap_size:]
        hint = word[:-gap_size]
        if gap_size >= len(word):
            logging.info(f"Error! Gap of size {gap_size} is larger than the word {word}.")
            # We need at least 1 character as a hint:
            gap = word[1:]
        
        new_features = {}
        new_features['compound_break'] = self.is_compound_break(word, gap)
        new_features['th_gap'] = self.get_th_gap(hint)
        new_features['syllable_break'] = self.is_syllable_break(word, gap)
        new_features['length_of_solution_chars'] = gap_size
        new_features['length_of_solution_sylls'] = self.get_num_syllables(word)
        
        if self.update_bert:
            new_features['bert_top50_entropy'] = self.get_bert_entropy_top50(word, gap)
            new_features['bert_gap_proba'] = self.get_bert_token_proba(word, gap)

        return word, gap, new_features
        
    def get_gap_idx(self, word: str, gap_size: int) -> (str, str, list):
        # Does the same as get_gap but replaces the keys with the appropriate idx in the feature list.
        word = word
        gap = word[-gap_size:]
        hint = word[:-gap_size]
        if gap_size >= len(word):
            # We need at least 1 character as a hint:
            gap = word[1:]
        if gap_size == 0:
            # We need at least 1 character as a gap:
            gap = word[-1]
        
        new_features = {}
        new_features[self.feature_index_dict['compound_break']] = self.is_compound_break(word, gap)
        new_features[self.feature_index_dict['th_gap']] = self.get_th_gap(hint)
        new_features[self.feature_index_dict['syllable_break']] = self.is_syllable_break(word, gap)
        new_features[self.feature_index_dict['length_of_solution_chars']] = gap_size
        new_features[self.feature_index_dict['length_of_solution_sylls']] = self.get_num_syllables(word)
        
        if self.update_bert:
            new_features[self.feature_index_dict['bert_top50_entropy']] = self.get_bert_entropy_top50(word, gap)
            new_features[self.feature_index_dict['bert_gap_proba']] = self.get_bert_token_proba(word, gap)

        return word, gap, new_features

    def get_feature_space(self, word: str) -> dict:
        # Fetches the whole possible feature space
        # returns an n*m array where n are the gap positions and m the according features
        feature_space = {}
        for i in range(1, len(word)): # we need at least one gap or one hint char
            gap = word[i:]
            word, gap, feats = self.get_gap(word, len(gap))
            feature_space[gap] = feats
        
        return feature_space
       
    def get_feature_space_idx(self, word: str) -> dict:
        # Fetches the whole possible feature space
        # returns an n*m array where n are the gap positions and m the according features
        feature_space = {}
        for i in range(1, len(word)): # we need at least one gap or one hint char
            gap = word[i:]
            word, gap, feats = self.get_gap_idx(word, len(gap))
            feature_space[gap] = feats

        return feature_space
        
    def get_feature_space_idx_gap_size(self, word: str) -> dict:
        # Fetches the whole possible feature space
        # returns an n*m array where n are the gap positions and m the according features
        feature_space = {}
        for i in range(1, len(word)): # we need at least one gap or one hint char
            gap = word[i:]
            word, gap, feats = self.get_gap_idx(word, len(gap))
            feature_space[len(gap)] = feats

        return feature_space
        
""" Class that manipulates only on the position level.
    For a sentence 
"""    
class PositionManipulator:
    def __init__(self) -> None:
        self.feature_index_dict = {
                                   "gaps_in_sent": 51,
                                   "preceding_gaps": 52,
                                   "preceding_gaps_sent": 53,
                                   "occurs_as_gap": 54,
                                   #"gap_token_index": 55,
                                   }

    def get_bounds(self, data: dict) -> dict:
        results = {}
        filtered = self.filter_empty(data)
        for key, vals in filtered.items():
            gap_idx, sent_idx = key
            results[key] = {51:self.get_max_nr_gaps_in_sent(filtered, sent_idx),
                            52:gap_idx, # Max pred gaps is same as the gap idx
                            53:self.get_max_nr_pred_gaps_sent(filtered, gap_idx, sent_idx),
                            54:self.get_max_occurs_as_gap(filtered, gap_idx),
                            #55:self.get_position_of_gap(gap_idx)
                            }
        return results
            
        
    def initialize_c_test(self) -> dict:
        # Initializes the set gaps in the current c-test.
        # Filter tokens that are index in gap_idx.
        c_test = {}
        gap_count = 0
        idx_count = 0
        for idxs, vals in sorted(self.data.items()):
            tok_idx, sent_idx = idxs
            if idx_count in self.gap_idx:
                c_test[(tok_idx, sent_idx)] = vals
                gap_count += 1
            idx_count += 1
        try:
            assert(gap_count == 20)
        except AssertionError:
            logging.info(f"Error! Expected 20 gaps but got {gap_count}!")
            
        return c_test
        
    def filter_empty(self, data: dict) -> dict:
        results = {}
        for idx, vals in data.items():
            if len(vals["features"]) < 2:
                continue
            results[idx] = vals
        
        return results
                
    def initialize_word_lookup(self) -> dict:
        # Lookup dictionary that maps words to their occurring indices
        words = {}
        for idxs, vals in self.data.items():
            try:
                words[vals["word"]].append(idxs)
            except KeyError:
                words[vals["word"]] = [idxs]
        return words
        
    def get_max_nr_gaps_in_sent(self, data: dict, sent_idx: int) -> float:
        # Number of gaps in the given sentence
        count = [1.0 for i in data.keys() if sent_idx in i]
        return sum(count)
    
    def get_max_nr_pred_gaps_sent(self, data: dict, gap_idx: int, sent_idx: int) -> float:
        # Number of preceeding gaps in the same sentence given a specific gap
        pred_sent_count = [1.0 for i in data.keys() if i[0] < gap_idx and sent_idx in i]
        return sum(pred_sent_count)
    
    def get_max_occurs_as_gap(self, data: dict, gap_idx: int) -> float:
        # Checks of the word also occurs as a gap elsewhere in the C Test
        words = {}
        for idxs, vals in data.items():
            try:
                words[vals["word"]].append(idxs)
            except KeyError:
                words[vals["word"]] = [idxs]
                
        word = [x["word"] for i, x in data.items() if gap_idx in i]
        if len(words[word[0]]) > 1:
            return 1.0
        return 0.0

    def get_nr_gaps_in_sent(self, sent_idx: int) -> float:
        # Number of gaps in the given sentence
        count = [1.0 for i in self.c_test.keys() if sent_idx in i]
        return sum(count)

    def get_nr_pred_gaps(self, gap_idx: int) -> float:
        # Number of preceeding gaps given a specific gap
        pred_count = [1.0 for i in self.c_test.keys() if gap_idx < i[0]]
        return sum(pred_count)
    
    def get_nr_pred_gaps_sent(self, gap_idx: int, sent_idx: int) -> float:
        # Number of preceeding gaps in the same sentence given a specific gap
        pred_sent_count = [1.0 for i in self.c_test.keys() if gap_idx < i[0] and sent_idx in i]
        return sum(pred_sent_count)
    
    def get_occurs_as_gap(self, gap_idx: int) -> float:
        # Checks of the word also occurs as a gap elsewhere in the C Test
        word = [x["word"] for i, x in self.c_test.items() if gap_idx in i]
        if len(self.word_dict[word[0]]) > 1:
            return 1.0
        return 0.0
        
    def get_occurs_as_gap_matrix(self, data: dict) -> dict:
        # Fetch the keys with similar words for each word
        results = {k:[] for k in data.keys()}
        words = set([x["word"].lower() for x in data.values()])
        word_lookup = {w:[] for w in words}
        for k,v in data.items():
            word_lookup[v["word"].lower()].append(k)
        for k,v in data.items():
            tmp_words = word_lookup[v["word"].lower()].copy()
            tmp_words.remove(k)
            results[k] = tmp_words
        return results
    
    def get_position_of_gap(self, gap_idx: int) -> float:
        # Get the token index of the gap (equals the gap index)
        return float(gap_idx)
    
    def get_pos(self, data: dict, gap_idx: list) -> dict:
        # Returns a dict with all changed features for a list of input gap_idxs
        self.data = self.filter_empty(data) 
        self.gap_idx = gap_idx
        self.c_test = self.initialize_c_test()
        self.word_dict = self.initialize_word_lookup()      

        result = {}
        
        for idxs, val in self.c_test.items():
            gap_idx, sent_idx = idxs
            new_features = {}
            new_features['gaps_in_sent'] = self.get_nr_gaps_in_sent(sent_idx)
            new_features['preceding_gaps'] = self.get_nr_pred_gaps(gap_idx)
            new_features['preceding_gaps_sent'] = self.get_nr_pred_gaps_sent(gap_idx, sent_idx)
            new_features['occurs_as_gap'] = self.get_occurs_as_gap(gap_idx)
            #new_features['gap_token_index'] = self.get_position_of_gap(gap_idx)
            result[idxs] = new_features
        
        return result
        
    def get_pos_idx(self, data: dict, gap_idx: list) -> dict:
        # Returns a dict with all changed features for a list of input gap_idxs but with indexes
        self.data = self.filter_empty(data)
        self.gap_idx = gap_idx
        self.c_test = self.initialize_c_test()
        self.word_dict = self.initialize_word_lookup()      

        result = {}
        
        for idxs, val in self.c_test.items():
            gap_idx, sent_idx = idxs
            
            new_features = {}
            new_features[self.feature_index_dict['gaps_in_sent']] = self.get_nr_gaps_in_sent(sent_idx)
            new_features[self.feature_index_dict['preceding_gaps']] = self.get_nr_pred_gaps(gap_idx)
            new_features[self.feature_index_dict['preceding_gaps_sent']] = self.get_nr_pred_gaps_sent(gap_idx, sent_idx)
            new_features[self.feature_index_dict['occurs_as_gap']] = self.get_occurs_as_gap(gap_idx)
            result[idxs] = new_features
            
        return result

    def get_pos_features(self, data: dict, gap_idx: list) -> dict:
        # Returns a dict with all changed features for a list of input gap_idxs but with indexes
        # NOTE: in contrast to the other functions, this returns a dictionary with all feature vectors!
        self.data = self.filter_empty(data)
        self.gap_idx = gap_idx
        self.c_test = self.initialize_c_test()
        self.word_dict = self.initialize_word_lookup()      

        result = {}
        
        for idxs, val in self.c_test.items():
            gap_idx, sent_idx = idxs
            feature = val["features"]
            feature[self.feature_index_dict['gaps_in_sent']] = self.get_nr_gaps_in_sent(sent_idx)
            feature[self.feature_index_dict['preceding_gaps']] = self.get_nr_pred_gaps(gap_idx)
            feature[self.feature_index_dict['preceding_gaps_sent']] = self.get_nr_pred_gaps_sent(gap_idx, sent_idx)
            feature[self.feature_index_dict['occurs_as_gap']] = self.get_occurs_as_gap(gap_idx)
            
            result[idxs] = {"features":feature, 
                               "word":val["word"], 
                               "gap":val["gap"],
                               "hint":val["hint"]
                               }
            
        return result

        
