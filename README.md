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
