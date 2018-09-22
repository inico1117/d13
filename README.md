# d13
#GridSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X = iris.data
y = iris.target
k_range = range(1,31)
knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(X,y)
grid.cv_results_
print(grid.best_score_) => 0.98
print(grid.best_params_) => {'n_neighbors': 13}
print(grid.best_estimator_) => KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=13, p=2,
                               weights='uniform')
weight_options = ['uniform','distance']
param_grid = dict(n_neighbors=k_range,weights=weight_options)
grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(X,y)
print(grid.best_score_) => 0.98
print(grid.best_params_) => {'n_neighbors': 13, 'weights': 'uniform'}

#RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
param_dist = dict(n_neighbors=k_range,weights=weight_options)
rand = RandomizedSearchCV(knn,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=5)
rand.fit(X,y)
#print(rand.cv_results_)
print(rand.best_score_)
print(rand.best_params_)
