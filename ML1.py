# train a classifier

# Supervised Learning: Create a classifier by finding patterns in examples

'''
Supervised Learning Recipe:

Collect Training Data --> Train Classifier --> Make Prediction 

Training Data

Weight | Texture | Label
-------+---------+--------
 150g  |  Bumpy  | Orange
-------+---------+--------
 170g  |  Bumpy  | Orange
-------+---------+--------
 140g  | Smooth  | Apple
-------+---------+--------
 130g  | Smooth  | Apple
-------+---------+--------
 ....  |  .....  | .....

'''


from sklearn import tree

# features is the input to the classifier (smooth = 1; bumpy = 0)
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# labels is the output that we want (apple = 0; orange = 1)
labels = [0, 0, 1, 1]  

clf = tree.DecisionTreeClassifier()

# fit() - it's a learning algorithm that finds patterns in our training data
clf = clf.fit(features, labels)

print(clf.predict([[160, 0]]))
