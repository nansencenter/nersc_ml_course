import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor,export_graphviz
import graphviz
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

dpi = 200


boston = datasets.load_boston()
X = boston.data
y = boston.target

reg = DecisionTreeRegressor(random_state=0,max_depth=3)
reg = reg.fit(X,y)
rf_boston = RandomForestRegressor(n_estimators=1000,max_features=10,random_state=10)
rf_boston.fit(X,y)
dot_data = export_graphviz(reg, out_file=None, max_depth=3,feature_names=boston.feature_names)
graph = graphviz.Source(dot_data,format='png')
graph.render('tree')


# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = RandomForestRegressor(n_estimators=200,max_depth=6, oob_score=True, min_samples_leaf=2, criterion='mae')

regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)
# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)
# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="lightgray", label="data")
plt.xlabel("data")
plt.ylabel("target")
plt.legend()
plt.savefig('tree-data.png', dpi=dpi,bbox_inches = 'tight', pad_inches = 0)


plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.legend()
plt.savefig('tree-2.png', dpi=dpi,bbox_inches = 'tight', pad_inches = 0)

plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.legend()
plt.savefig('tree-5.png', dpi=dpi,bbox_inches = 'tight', pad_inches = 0)

plt.plot(X_test, y_3, color="tomato", label="RF", linewidth=2)
plt.legend()
plt.savefig('RF.png', dpi=dpi,bbox_inches = 'tight', pad_inches = 0)


plt.show()

dot_data = export_graphviz(regr_1, out_file=None, max_depth=3,feature_names=['data'])
graph = graphviz.Source(dot_data,format='png')
graph.render('tree_uni_2')

dot_data = export_graphviz(regr_2, out_file=None, max_depth=3,feature_names=['data'])
graph = graphviz.Source(dot_data,format='png')
graph.render('tree_uni_5')

importances = rf_boston.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_boston.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
firsts = 5
plt.title("Feature importances")
plt.bar(range(firsts), importances[indices][:firsts],
       color="r", yerr=std[indices][:firsts], align="center")
plt.xticks(range(firsts), boston.feature_names[indices][:firsts])
plt.xlim([-1, firsts])
plt.savefig('importance_RF.png', dpi=dpi,bbox_inches = 'tight', pad_inches = 0)

plt.show()