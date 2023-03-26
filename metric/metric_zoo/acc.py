from sklearn.metrics import accuracy_score


class Accuracy:
    def __init__(self):
        self.func = accuracy_score

    def __call__(self, y_pred, y_true):
        return {'acc': accuracy_score(y_true, y_pred)}

        


