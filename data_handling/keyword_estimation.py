import nltk
from collections import Counter
import torch 
from re import sub

def keyword_estimator(caption, vocab):
    potential_keywords = []
    sentence = ' '.join(caption)
    sentence = sentence.lower()
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')
    sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    words = sentence.strip().split()
    pos_tag = nltk.pos_tag(words)
    
    
    

    for word, pos in pos_tag:

        if (pos.startswith("V") or pos.startswith("N")) and word in vocab:
            potential_keywords.append(word)
    
    keywords_counter = Counter(potential_keywords)

    top_keywords = [vocab.index(word) for word,_ in keywords_counter.most_common(5)]
    print(top_keywords)
    # top_keywords = ' '.join(top_keywords)
    top_keywords = torch.tensor(top_keywords, dtype=torch.long)
    return top_keywords



        
        
    





    

