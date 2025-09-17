import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge 
import sys

#https://aldosari.medium.com/evaluating-text-quality-with-bleu-score-in-python-using-nltk-f45f3b16c8e0
#
def compute_bleu(reference_text, candidate_text):
    reference = [reference_text.split()]
    candidate = candidate_text.split()
    
    # Compute BLEU score
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score

#https://pypi.org/project/rouge/
def compute_rouge(reference_text, candidate_text):
    #sys.setrecursionlimit(len(reference_text.split()) * len(candidate_text.split()) + 10)
    max_length = 800  # Truncate texts to a maximum of 500 tokens
    truncated_reference = ' '.join(reference_text.split()[:max_length])
    truncated_candidate = ' '.join(candidate_text.split()[:max_length])
    rouge = Rouge()
    scores = rouge.get_scores(truncated_candidate, truncated_reference)
    return scores

def compute_jaccard_index(reference_text, candidate_text):
    reference = set(reference_text.split())
    candidate = set(candidate_text.split())

    # Compute intersection and union
    intersection = reference.intersection(candidate)
    union = reference.union(candidate)
    
    # Jaccard Index formula
    return len(intersection) / (len(union) + 0.001)