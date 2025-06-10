import torch
import sys

checkpoint = torch.load("cbow_text8.pt")
embeddings = checkpoint["embeddings"]
word_to_ix = checkpoint["word_to_ix"]
ix_to_word = checkpoint["ix_to_word"]


# Test semantic similarity
def find_similar(word, top_k=5):
    if word not in word_to_ix:
        return f"'{word}' not in vocabulary"

    idx = word_to_ix[word]
    word_vec = embeddings[idx]

    # Cosine similarity
    similarities = torch.cosine_similarity(word_vec.unsqueeze(0), embeddings)
    top_indices = similarities.topk(top_k + 1)[1][1:]  # Skip the word itself

    return [ix_to_word[i.item()] for i in top_indices]

def main(word):
    print(f"{word}: " + ", ".join(find_similar(word)))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <word>")
        sys.exit(1)
    main(sys.argv[1])