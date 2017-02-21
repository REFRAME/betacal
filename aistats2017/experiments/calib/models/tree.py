from sklearn.tree import DecisionTreeClassifier

class MyDecisionTreeClassifier(DecisionTreeClassifier):
    def predict_proba(self,X):
        alpha = 1
        leaf_indices = self.apply(X)
        leaf_counts = self.tree_.value[leaf_indices]
        leaf_counts = leaf_counts.squeeze()
        leaf_counts += alpha
        return leaf_counts/leaf_counts.sum(axis=1).reshape(-1,1)
