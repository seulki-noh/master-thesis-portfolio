import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def loocv_dt_classifier_then_cluster_mean(df, feature_cols, cluster_col, target_col, max_depth=2, random_state=2):
    errors = []
    X_full = df[feature_cols]
    y_cluster = df[cluster_col].values
    y = df[target_col].values
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X_full):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_cluster = y_cluster[train_idx]
        y_train = y[train_idx]
        true_y = y[test_idx[0]]

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        clf.fit(X_train, y_train_cluster)

        pred_cluster = clf.predict(X_test)[0]
        cluster_targets = y_train[y_train_cluster == pred_cluster]
        if len(cluster_targets) == 0:
            continue

        pred_y = cluster_targets.mean()
        errors.append(abs(pred_y - true_y))

    return float(np.mean(errors)) if errors else np.nan


def loocv_dt_regressor(df, feature_cols, target_col, max_depth=2, random_state=2):
    errors = []
    X_full = df[feature_cols]
    y = df[target_col].values
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X_full):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train = y[train_idx]
        true_y = y[test_idx[0]]

        reg = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        reg.fit(X_train, y_train)

        pred_y = reg.predict(X_test)[0]
        errors.append(abs(pred_y - true_y))

    return float(np.mean(errors)) if errors else np.nan