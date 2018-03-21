from sklearn import neighbors as nb
nb_neighb = 25
clf = nb.KNeighborsClassifier(nb_neighb)
X=
clf.fit(X,Y)
print(clf_predict(X[0:5]))
