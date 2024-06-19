import json
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model

# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Get word vector from GloVe embeddings
def get_word_vector(word, embeddings_index, embedding_dim=50):
    return embeddings_index.get(word.lower(), np.zeros(embedding_dim))

def solve_puzzle(words, model, embeddings_index, embedding_dim=50):
    if len(words) != 16:
        raise ValueError('Puzzle must contain exactly 16 words')
    X_puzzle = np.array([get_word_vector(word, embeddings_index, embedding_dim) for word in words])
    predictions = model.predict(X_puzzle)
    predicted_categories = np.argmax(predictions, axis=1)
    
    # Use K-means clustering to group words
    kmeans = KMeans(n_clusters=4, random_state=0)
    clusters = kmeans.fit_predict(X_puzzle)
    
    # Collect words in each cluster
    cluster_groups = [[] for _ in range(4)]
    for word, cluster in zip(words, clusters):
        cluster_groups[cluster].append(word)

    # Initialize solution with empty lists for each group
    solution = {f'Group {i + 1}': [] for i in range(4)}

    # Distribute words from clusters to groups
    for i, group in enumerate(cluster_groups):
        group_name = f'Group {i + 1}'
        for word in group:
            added = False
            for j in range(4):
                if len(solution[f'Group {j + 1}']) < 4:
                    solution[f'Group {j + 1}'].append(word)
                    added = True
                    break
            if not added:
                # If all groups have exactly 4 words, place extra words in a new group
                solution[f'Group {len(solution) + 1}'] = [word]

    # Ensure all groups have exactly 4 words
    for i in range(4):
        while len(solution[f'Group {i + 1}']) < 4:
            solution[f'Group {i + 1}'].append('')  # Append empty string to fill up

    return solution



if __name__ == "__main__":
    # Load GloVe embeddings
    glove_file = 'data/glove.6B.50d.txt'
    embeddings_index = load_glove_embeddings(glove_file)
    embedding_dim = 50  # GloVe 50-dimensional embeddings

    # load the trained model
    model = load_model('models/nyt_connections_model.h5')

    # load label encoder classes
    with open('models/label_encoder.json', 'r') as f:
        label_encoder_classes = json.load(f)

    puzzle_words = ['apple', 'orange', 'banana', 'grape', 'red', 'yellow', 'green', 'purple', 'one', 'two', 'three', 'four', 'square', 'circle', 'triangle', 'rectangle']
    solution = solve_puzzle(puzzle_words, model, embeddings_index, embedding_dim)
    print(solution)
