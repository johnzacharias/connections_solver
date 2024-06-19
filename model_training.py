import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

# Load the JSON data
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Preprocess data for machine learning
def preprocess_data(data):
    words = []
    categories = []

    for game in data:
        for answer in game['answers']:
            category = answer['group']
            for word in answer['members']:
                words.append(word)
                categories.append(category)

    # Encode categories
    label_encoder = LabelEncoder()
    categories_encoded = label_encoder.fit_transform(categories)

    return words, categories_encoded, label_encoder

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/connections.json')
    words, categories_encoded, label_encoder = preprocess_data(data)

    # Load GloVe embeddings
    glove_file = 'data/glove.6B.50d.txt'
    embeddings_index = load_glove_embeddings(glove_file)
    embedding_dim = 50  # GloVe 50-dimensional embeddings

    # Prepare X (word vectors) and y (encoded categories)
    X = np.array([get_word_vector(word, embeddings_index, embedding_dim) for word in words])
    y = to_categorical(categories_encoded)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential()
    model.add(Dense(128, input_dim=embedding_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save('models/nyt_connections_model.h5')
    with open('models/label_encoder.json', 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)
