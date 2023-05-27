import gensim.downloader as api


def main():
    wv = api.load("word2vec-google-news-300")

    # We can easily obtain vectors for terms the model is familiar with:
    vec_king = wv["king"]

    # Print the 5 most similar words to king
    wv.most_similar(positive=["car"], topn=5)
