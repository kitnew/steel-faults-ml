from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def get_svm(kernel='rbf', C=1.0, class_weight='balanced', seed=5296):
    return SVC(kernel=kernel, C=C, class_weight=class_weight, random_state=seed)

def get_random_forest(n_estimators=200, max_depth=10, class_weight='balanced', seed=5296):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight, random_state=seed)

def get_gradient_boosting(loss='log_loss', learning_rate=0.1, n_estimators=200, max_depth=10, seed=5296):
    return GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=seed)

def get_knn(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def get_log_reg(max_iter=1000, class_weight='balanced', seed=5296):
    return LogisticRegression(max_iter=max_iter, class_weight=class_weight, random_state=seed)