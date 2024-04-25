import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bc_df = pd.read_csv('./breast-cancer-wisconsin.data', header=None)
bc_df.replace('?', np.nan, inplace=True)
bc_df.dropna(inplace=True)

model = DecisionTreeClassifier()

X = bc_df.iloc[:, 1:-1]
y = bc_df.iloc[:, -1]

y = y.replace({2: 0, 4: 1})
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(x_train, y_train)
result = model.predict(x_test)
acc = accuracy_score(y_test, result)

print(acc)
print(X.shape)