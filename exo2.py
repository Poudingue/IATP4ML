import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import random

data = pd.read_csv("full_dataset.csv")

features_cols = ["my_health", "opponent_health", "my_hand", "opponent_hand", "my_board_nb_creatures",
     "my_board_total_health", "my_board_total_attack", "opponent_board_nb_creatures",
     "opponent_board_total_health", "opponent_board_total_attack", "me_playing"]

pred_col = ["result"]

turns_before_end = 4

turn_df = data[data["turns_to_end"] == turns_before_end]
X = turn_df[features_cols]
Y = turn_df[pred_col]

from sklearn import neighbors as nb
nb_neighb = 25
clf = nb.KNeighborsClassifier(n_neighbors=nb_neighb)

clf.fit(X, Y)
print(clf.predict(X[0:5]))
