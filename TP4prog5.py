from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import random

data = pd.read_csv("full_dataset.csv")

features_cols = ["my_health", "opponent_health", "my_hand", "opponent_hand", "my_board_nb_creatures",
     "my_board_total_health", "my_board_total_attack", "opponent_board_nb_creatures",
     "opponent_board_total_health", "opponent_board_total_attack", "me_playing"]

pred_col = ["result"]

turns_before_end = 4

turn_df = data[data["turns_to_end"] == turns_before_end]
X = turn_df[features_cols]
Y = turn_df[pred_col]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=.3, random_state=random.seed())

for i in range(1,51):
    clf=tree.DecisionTreeClassifier(max_leaf_nodes=i*50)
    clf=clf.fit(Xtrain,Ytrain)
    print("i="+str(i)+" : "+str(clf.score(Xtest,Ytest)))
