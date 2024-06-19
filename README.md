CONNECTIONS SOLVER - John Zacharias

This project aims to group an array of 16 words into four categories of four words each, as per the rule of the popular NYT puzzle game Connections. 
The model was trained on a json dataset of past Connections games using the Stanford 50 dimensional GloVe embeddings as features. It then uses k-means clustering to group the words into distinct groups. 

How to Run

-Clone Repository

-Install Requirements

- Download 50 dimensional GloVe file (from Stanford site or Kaggle)

-Run solve_puzzle.py after replacing the array of words with your own array of 16 words
