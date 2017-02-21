import pyGPs

class myGPC(pyGPs.gp.GPC):
    def __init__(self):
        super(myGPC, self).__init__()

    def fit(self, train_x, train_y):
        train_y = train_y*2-1
        super(myGPC, self).getPosterior(train_x, train_y)
        super(myGPC, self).optimize(train_x, train_y)

    def predict_proba(self, x):
        ym, ys2, fm, fs2, lp = self.predict(x)
        return ym

    def predict(self, x):
        ym, ys2, fm, fs2, lp = super(myGPC, self).predict(x)
        ym = (ym+1)/2
        ys2 = ys2/2
        return ym, ys2, fm, fs2, lp


