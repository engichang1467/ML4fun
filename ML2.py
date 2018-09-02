# visualize decision tree
'''

Iris - classic ML problem, we want to identify what type of flower you have based on different measurements like the lenght & width of the petal

The data set includes three different types of flowers
They're all species of iris -- setosa, versicolor, and virginica

Goals
1. import dataset.
2. Train a classifier
3. Predict label for new flower.
4. Visualize the tree


Testing Data
* Examples used to "test" the classifier's accuracy
* Not part of the training data.

Just like in programming, testing is a very important part of ML

'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
'''
print(iris.feature_names)
print(iris.target_names)
# The features & examples themselves are contained in the data varaible
print(iris.data[0])
# The target variable contains the labels
print(iris.target[0])  # 0 = setosa


# we can print out the entire dataset like this
for i in range(len(iris.target)):
	print("example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

'''

test_idx = [0,50,100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Predict label for new flower
print(test_target)
print(clf.predict(test_data))

# vis code
'''
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
						out_file=dot_data,
						feature_names=iris.feature_names,
						class_names=iris.target_names,
						filled=True, rounded=True,
						impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
'''
print(test_data[1], test_target[1])

print(iris.feature_names, iris.target_names)
