from sklearn import tree
# -*- coding: utf-8 -*-
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

for i in range(3,10):
    clf=tree.DecisionTreeClassifier(max_leaf_nodes=i)
    clf=clf.fit(X,Y)
    print(clf.predict(X[0:10]))
    print(clf.score(X,Y))
    tree.export_graphviz(clf, out_file='tree'+str(i)+'.dot')
