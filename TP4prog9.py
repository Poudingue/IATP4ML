from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import random
import pandas as pd

data = pd.read_csv("full_dataset.csv")

features_cols = ["my_health", "opponent_health", "my_hand", "opponent_hand", "my_board_nb_creatures",
     "my_board_total_health", "my_board_total_attack", "opponent_board_nb_creatures",
     "opponent_board_total_health", "opponent_board_total_attack", "me_playing"]

pred_col = ["result"]
print()
print("Réseau neuronal, 5 couches cachées de 5 neurones :")
print()
for i in range(1, 20):
    turns_before_end = i

    turn_df = data[data["turns_to_end"] == turns_before_end]
    X = turn_df[features_cols]
    Y = turn_df[pred_col]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=.3, random_state=1)

    clf=MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5,5), random_state=1)
    clf.fit(Xtrain,Ytrain)
    print("Score "+str(i)+" tours avant la fin : \t"+str(clf.score(Xtest, Ytest)))
print()
print("Arbre de décisions de maximum 5 de profondeur :")
print()
for i in range(1,20):
    turns_before_end = i

    turn_df = data[data["turns_to_end"] == turns_before_end]
    X = turn_df[features_cols]
    Y = turn_df[pred_col]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=.3, random_state=1)

    clf=tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
    clf=clf.fit(Xtrain,Ytrain)
    print("Score "+str(i)+" tours avant la fin : \t"+str(clf.score(Xtest,Ytest)))
