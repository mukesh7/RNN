from nltk.translate.bleu_score import sentence_bleu

"""
Calculates the Bleu Score of two given files
"""
def scoring_func(ref, pre):
    reference = open(ref, 'r').readlines()
    candidate = open(pre, 'r').readlines()
    if len(reference) != len(candidate):
    	raise ValueError('The number of sentences in both files do not match.')
    
    score = 0.
    
    for i in range(len(reference)):
    	score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())
    
    score /= len(reference)
    print("The bleu score is: "+str(score))
    return score
