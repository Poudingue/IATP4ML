# IA TP4 : Machine Learning avec scikit-learn

## Exercice 1

```
print(len(data))
```
Affiche la taille des datas, en nombre de lignes


```
print(data.columns[0])
```
Affiche le nom de la colonne 0

```
print(data.iloc[0])
```
Affiche les infos de la ligne de données numéro 0

```
print(data[0:5])
```
Affiche les données des parties 0 à 5 exclue

```
print(data["my_health"])
```
Affiche la colonne ayant l'intitulé «my_health»

```
print(data[["my_health","my_hand"]][0:5])
```
Affiche les colonnes «my_health» et «my_hand» pour les 5 premières lignes

```
turn_df = data[data["turns_to_end"]==turns_before_end]
```
Entre dans turn_df l'ensemble des lignes telles que «turn_to_end» corresponde à la variable turn_to_end telle que définie précédemment

```
plt.scatter(X[x_col], X[y_col], c=Y)
plt.show
```
Crée un graphe prenant en abcisse X[x_col] et en ordonnée X[y_col], (my_health et opponent_health).
c=Y attribue des valeurs aux points en fonction de si la partie est gagnée/perdue.
plt.show() affiche le graphe.

```
plt.xlabel(x_col)
```
Étiquette l'axe x avec x_col (my_health)

```
print(X[Y["result"]==0])
plt.scatter(X[Y["result"]==0][x_col], X[Y["result"]==0][y_col], color="red", label="lose")
plt.legend()
```
Permet de définir plus précisémment l'affichage du graphe. Les result==0 servent à repérer les parties qu'on a perdu, et le color="red", ainsi que label="lose" permettent de légender correctement les points, par des couleurs que l'on a choisies

### À quoi sert le random ?
Le random sert à éviter d'avoir des points superposés à cause de la vie qui est une donnée discrète, et permet donc de voir plusieurs points de données plutôt que juste le dernier qui a été ajouté sur la pile.

## Exercice 2

```
clf.fit(X,Y)
```
Entraîne le classifieur clf pour prévoir les données Y à partir des données X

```
print(clf.predict_proba(X[0:5]))
```
Affiche les probabilités que la prédiction soit correcte

```
print(clf.score(X,Y))
```
Affiche la précision moyenne du modèle pour prédire Y depuis X.

### Expliquer pourquoi l'explication peut être pessimiste
Cette évaluation peut être pessimiste car elle est entraînée sur moins d'échantillons, alors que le modèle réel final sera entraîné sur tous les échantillons.

```
len(X_train[Y_train["result"]==0])
```
Affiche la quantité de cas où result == 0 (partie perdue) dans l'échantillon d'entraînement.

```
from sklearn.metrics import confusion_matrix
```
Import de confusion_matrix, qui comme son nom l'indique permet de créer une matrice de confusion (affichage des vrais/faux positifs/négatifs) afin d'évaluer la qualité de notre modèle.

### Expliquer en quelques lignes ce que l'on voit (en général) dans une matrice de confusion.

Dans une matrice de confusion, on a un tableau classe estimée/classe réelle. Si le modèle est à peu près correct, la quantité dans les cases partageant leur classe estimée est réelle est supérieure à la quantité ayant des classes estimées et réelles différentes.

|         | estimé positif | estimé négatif |
|---------|----------------|----------------|
| positif | beaucoup       | peu            |
| négatif | peu            | beaucoup       |

### Que se passe-t-il avec shuffle=False ?
Si on n'active pas shuffle, les jeux d'apprentissage et de test sont tout le temps les mêmes.

### Que se passe-t-il si on met n_splits à 3 au lieu de 10 et shuffle à False ?

## Exercice 3

```
clf=tree.DecisionTreeClassifier()
```
Crée un classifieur de type arbre de décision

```
clf=clf.fit(X,Y)
```
Entraîne l'arbre de décision à prévoir Y en fonction de X

```
print(clf.predict(X[0:10]))
```
Affiche la prédiction pour les 10 premières valeurs de X

```
print(clf.score(X,Y))
```
Affiche le score de la prédiction

```
tree export_graphviz
```
Exporte un fichier graphviz représentant l'arbre de décision.

### Combien de feuilles contient l'arbre par défaut ?
Beaucoup trop. En diminuant le nombre de feuilles max, on a un arbre de plus en plus simple et petit, mais le score du classifieur diminue également. On a tout de même un score de 71,55% pour 3 feuilles

### Comparaison de Gini et entropy
gini obtient un score de classification légèrement meilleur, à quelques pourcents près, mais les résultats sont très similaires. La différence réside dans le choix des branches à créer. Les arbres un peu plus grands créés par analyse de l'entropie sont plus déséquilibrés, alors que les arbres créés par analyse du coeff de gini sont plutôt équilibrés.

Si l'objectif est de créer un arbre qui permet de décider le plus vite possible du résultat, choisir gini permet d'avoir un arbre moins profond pour le même nombre de feuilles, et d'arriver au résultat plus vite.

### Que remarquez-vous ? Quel est le nom de ce phénomène ?
Le fait de séparer les données d'entrainement et de test permettent d'avoir une meilleure estimation du score réel du classifieur.
En augmentant le nombre de feuilles autorisées, on se retrouve avec des scores plus bas. C'est de l'over-fitting, out surapprentissage, on essaye trop de suivre les données d'entraînement.

### Même chose avec la profondeur
On remarque exactement la même chose, autoriser une trop grande profondeur fait chuter le score à cause du surapprentissage.

## Exercice 4

```
clf=MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
clf.fit(X,Y)
```
On crée un classifieur de type bfgs, avec 2 couches cachées de 5 neurones chacune et on l'entraîne

### En faisant varier nb_neurons, nb_layers, taux d'apprentissage…
(Lancer exo4-split.py pour voir les résultats)
Le score semble globalement augmenter avec le nombre de couches cachées et le nombre de neurones par couche, mais le résultat est très chaotique, avec des scores faisant des hauts et des bas.
Le learning rate n'a pas l'air de changer quoi que ce soit à notre classifieur…
Peut-être que le classifieur est mal configuré.

## Exercice 5

### Quel classifieur est le plus adapté ?
L'arbre de décision me paraît très bon en rapport score/performance. Je pense personnellement que le réseau neuronal a moyen d'être très puissant sur ce type de problèmes, mais en l'entraînant plus longtemps, avec une configuration adaptée. Ici, on a des résultats très mauvais, peut-être à cause d'une mauvaise configuration du réseau neuronal.

### Tester la précision des classifieurs en fonction du nb de tours avant la fin

Le programme TP4prog9.py teste la précision du réseau neuronal et de l'arbre de décision en fonction de la proximité avec la fin. Il est difficile de départager lequel des deux est le meilleur. Le réseau neuronal a parfois un score meilleur, mais il a également des cas où il est bien pire, il n'a pas la même constance dans son score et est donc potentiellement moins fiable
