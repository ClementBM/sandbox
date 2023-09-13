import re
import math

def parse_serie_name(serie_name:str):
    first_pattern = r"([^,:]+)"
    second_pattern = r"(\s{0,1}\([^\)]+\)|\s{0,1}\[[^\[]+\])"
    
    matches = re.compile(first_pattern).findall(serie_name)
    primary_matches =  [match.strip() for match in matches if match.strip() != ""]

    tokens = []
    for primary_match in primary_matches:
       tokens.append(re.sub(second_pattern, "", primary_match))
    
    return [token.replace("  ", " ") for token in tokens]

def get_dict_count(tokens):
    dict_count = {}
    for token in tokens:
        if token in list(dict_count.keys()):
            dict_count[token] += 1
        else:
            dict_count[token] = 1
            
    return dict(sorted(dict_count.items(), key=lambda item: item[1], reverse=True))

def get_dict_freq(dict_count:dict):
    token_count = sum(dict_count.values())
    return {k:(v/token_count) for k, v in dict_count.items()}

def get_doc_dicts(document_tokens):
    return [set(tokens) for tokens in document_tokens]

def get_idf_scores(corpus_tokens, document_dicts):
    idfs = []

    for token in corpus_tokens:
        document_token_occurences = sum([token in doc_dict for doc_dict in document_dicts])
        idf = math.log(len(document_dicts) / document_token_occurences)
        idfs.append(idf)
    return idfs

def get_tfidf_scores(doc_freqs, idf_scores):
    scores = []
    for tf, idf in zip(doc_freqs.values(), idf_scores):
        scores.append(tf * idf)
    
    token_tfidfs = {k:v for k, v in zip(doc_freqs.keys(), scores)}
    return dict(sorted(token_tfidfs.items(), key=lambda item: item[1], reverse=True))
