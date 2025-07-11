import Levenshtein as Lev
from typing import List
from silero_utils import normalize_text

def cer(prediction, target):
    """
    Computes the Letter Error Rate, defined as the edit distance.
    Arguments:
        prediction (string): space-separated sentence
        target (string): space-separated sentence
        lang (string): language
    """
    # prediction, target, = prediction.replace(' ', ''), target.replace(' ', '')
    return Lev.distance(prediction, target), len(target)

def cer_wo_space(prediction, target):
    # remove space and enter
    prediction = prediction.replace(' ', '').replace('\n', '')
    target = target.replace(' ', '').replace('\n', '')
    return Lev.distance(prediction, target), len(target)

def wer(prediction, target):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        prediction (string): space-separated sentence
        target (string): space-separated sentence
        lang (string): language
    """
    # build mapping of words to integers
    b = set(prediction.split() + target.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    prediction = [chr(word2char[w]) for w in prediction.split()]
    target = [chr(word2char[w]) for w in target.split()]

    return Lev.distance(''.join(prediction), ''.join(target)), len(target)

def calculate_error_cnt(prediction, target):
    prediction = normalize_text(prediction)
    target = normalize_text(target)

    char_err_cnt, char_target_cnt = cer(prediction, target)
    char_wo_space_err_cnt, char_wo_space_target_cnt = cer_wo_space(prediction, target)
    word_err_cnt, word_target_cnt = wer(prediction, target)
    if char_target_cnt == 0:
        char_target_cnt = 1
    if char_wo_space_target_cnt == 0:
        char_wo_space_target_cnt = 1
    if word_target_cnt == 0:
        word_target_cnt = 1

    return char_err_cnt, char_target_cnt, char_wo_space_err_cnt, char_wo_space_target_cnt, word_err_cnt, word_target_cnt

def calculate_error_rate(results: List):
    char_err_tot_cnt = 0
    char_target_tot_cnt = 0
    char_wo_space_err_tot_cnt = 0
    char_wo_space_target_tot_cnt = 0
    word_err_tot_cnt = 0
    word_target_tot_cnt = 0
    
    for char_err_cnt, char_target_cnt, char_wo_space_err_cnt, char_wo_space_target_cnt, word_err_cnt, word_target_cnt in results:
        char_err_tot_cnt += char_err_cnt
        char_target_tot_cnt += char_target_cnt
        char_wo_space_err_tot_cnt+=char_wo_space_err_cnt
        char_wo_space_target_tot_cnt+=char_wo_space_target_cnt
        word_err_tot_cnt += word_err_cnt
        word_target_tot_cnt += word_target_cnt

    char_err_rate = char_err_tot_cnt / char_target_tot_cnt
    char_wo_space_err_rate = char_wo_space_err_tot_cnt / char_wo_space_target_tot_cnt
    word_err_rate = word_err_tot_cnt / word_target_tot_cnt

    return char_err_rate, char_wo_space_err_rate, word_err_rate
