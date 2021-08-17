# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd

# iris = load_iris()
# iris_data = iris.data
# iris_label = iris.target

# print(iris_data[0:100])
# print(iris_label[0:100])

# iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
# iris_df['label'] = iris.target
# print(iris_df)
# iris_df.to_csv('iris_df.csv',sep=',',na_rep='NaN')
# X_train, X_test, Y_train, Y_test = train_test_split(iris_data[0:100], iris_label[0:100], test_size=0.2)

# dt_clf = DecisionTreeClassifier()
# dt_clf.fit(X_train,Y_train)

# pred = dt_clf.predict(X_test)
# print(accuracy_score(Y_test,pred))

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

zeros = np.zeros(55,dtype=np.int8)
ones =  np.ones(45,dtype=np.int8)
actual = np.concatenate((zeros,ones),axis=None)

zeros = np.zeros(47,dtype=np.int8)
ones =  np.ones(53,dtype=np.int8)
predicted = np.concatenate((zeros,ones),axis=None)
np.random.shuffle(predicted[37:57])

print("confusion matrix\n",confusion_matrix(actual,predicted))
print("accuracy\t:",accuracy_score(actual,predicted))
print("precision\t:",precision_score(actual,predicted))
print("recall\t\t:",recall_score(actual,predicted))
print("f1\t\t:",f1_score(actual,predicted))