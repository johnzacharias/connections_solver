import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# load the JSON data
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# preprocess data for machine learning
def preprocess_data(data):
    words = []
    categories = []

    for game in data:
        for answer in game['answers']:
            category = answer['group']
            for word in answer['members']:
                words.append(word)
                categories.append(category)

    # encode categories
    label_encoder = LabelEncoder()
    categories_encoded = label_encoder.fit_transform(categories)

    return words, categories_encoded, label_encoder

# dummy function
#TODO: Replace with actual word embeddings
def get_word_vector(word):
    # Example: Using random embeddings for illustration
    return np.random.rand(100)  # Assuming 100-dimensional embeddings

# function to solve puzzles
def solve_puzzle(words, model, label_encoder):
    X_puzzle = np.array([get_word_vector(word) for word in words])
    predictions = model.predict(X_puzzle)
    predicted_categories = np.argmax(predictions, axis=1)
    
    # use K-means clustering to group words
    kmeans = KMeans(n_clusters=4, random_state=0)
    clusters = kmeans.fit_predict(X_puzzle)
    
    # initialize solution dictionary
    solution = {f'Group {i+1}': [] for i in range(4)}
    
    # organize words into groups based on predicted categories and clusters
    for word, category, cluster in zip(words, predicted_categories, clusters):
        group = f'Group {cluster + 1}'  # Adjust to match 1-based indexing
        solution[group].append(word)

    return solution

if __name__ == "__main__":
    # load and preprocess data
    data = load_data('data/connections.json')
    words, categories_encoded, label_encoder = preprocess_data(data)

    # prepare X (word vectors) and y (encoded categories)
    X = np.array([get_word_vector(word) for word in words])
    y = to_categorical(categories_encoded)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define the model
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # save the trained model
    model.save('models/nyt_connections_model.h5')

    #TEST
    puzzle_words = ["HAIL", "RAIN", "SLEET", "SNOW", "BUCKS", "HEAT", "JAZZ", "NETS",
                    "OPTION", "RETURN", "SHIFT", "TAB", "KAYAK", "LEVEL", "MOM", "RACECAR"]
    solution = solve_puzzle(puzzle_words, model, label_encoder)
    print(solution)
