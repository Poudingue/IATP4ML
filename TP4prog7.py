from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt

digits = load_digits()
X=digits.data
Y=digits.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=.3, random_state=random.seed())

for i in range(1,10):
    clf=MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5,i), random_state=1)
    clf.fit(Xtrain,Ytrain)
    print("5 neurons, "+str(i)+" hidden layers : \t" + str(clf.score(Xtest,Ytest)))

print()

for i in range(1,10):
    clf=MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(i,5), random_state=1)
    clf.fit(Xtrain,Ytrain)
    print(str(i)+" neurons, 5 hidden layers : \t" + str(clf.score(Xtest,Ytest)))

print()

print("with 5 hidden layers of 5 neurons :")
for i in range(1,10):
    clf=MLPClassifier(solver="lbfgs", alpha=1e-5, learning_rate_init=i*0.001, learning_rate="constant", hidden_layer_sizes=(5,5), random_state=1)
    clf.fit(Xtrain,Ytrain)
    print("learning rate of "+str("%.3f" % (i*0.001))+" : \t" + str(clf.score(Xtest,Ytest)))
print()

print("with 5 hidden layers of 5 neurons :")
for i in range(1,10):
    clf=MLPClassifier(solver="lbfgs", alpha=i*i*1e-5, hidden_layer_sizes=(5,5), random_state=1)
    clf.fit(Xtrain,Ytrain)
    print("with alpha ="+str("%.5f" % (i*i*1e-5))+" : \t\t" + str(clf.score(Xtest,Ytest)))
