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

x_col="my_health"
y_col="opponent_health"

colors=["red","green"]

for i in range(2):
    indices = Y["result"]==i
    plt.scatter(
        X[indices][x_col] + .5*(random(X[indices].shape[0])-.5),
        X[indices][y_col] + .5*(random(X[indices].shape[0])-.5),
                color=colors[i],label=i,s=2)
plt.legend()
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title("Hearthstone data: health")
plt.show()

# Le random sert à éviter d'avoir des points superposés à cause de la vie qui est une donnée discrète, et permet donc de voir plusieurs points de données plutôt que de juste voir le dernier qui a été ajouté
