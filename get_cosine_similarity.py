from sklearn.metrics.pairwise import cosine_similarity
from get_bert_word_embedding import get_sentence_embedding,get_sentence_transformer_embedding
#from sentence_transormers import SentenceTransformer
def get_cosine_similarity(sentence1,sentence2):
    # Get embeddings
    #model = SentenceTransformer('sentence-transformer/all-mpnet-base-v2')
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)

    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    #print(f"Cosine Similarity: {similarity}")
    return similarity