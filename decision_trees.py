class DecisionTree:
    def __init__(self):
        self.tree = None

    def entropy(self, y):
        from math import log2
        labels = set(y)
        return -sum(y.count(label) / len(y) * log2(y.count(label) / len(y)) for label in labels)

    def best_split(self, X, y):
        best_gain, best_feature = 0, None
        base_entropy = self.entropy(y)
        n_features = len(X[0])

        for feature in range(n_features):
            values = set(row[feature] for row in X)
            for value in values:
                left_y = [y[i] for i in range(len(y)) if X[i][feature] == value]
                right_y = [y[i] for i in range(len(y)) if X[i][feature] != value]
                split_entropy = (len(left_y) / len(y)) * self.entropy(left_y) + (len(right_y) / len(y)) * self.entropy(right_y)
                info_gain = base_entropy - split_entropy
                if info_gain > best_gain:
                    best_gain, best_feature = info_gain, (feature, value)

        return best_feature

    def build_tree(self, X, y):
        if len(set(y)) == 1:
            return y[0]

        feature, value = self.best_split(X, y)
        if feature is None:
            return max(set(y), key=y.count)

        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] == value:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return {(feature, value): {"Yes": self.build_tree(left_X, left_y), "No": self.build_tree(right_X, right_y)}}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        feature, value = list(tree.keys())[0]
        branch = "Yes" if x[feature] == value else "No"
        return self.predict_one(x, tree[feature, value][branch])

    def predict(self, X):
        return [self.predict_one(x, self.tree) for x in X]
