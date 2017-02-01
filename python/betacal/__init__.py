from .beta_calibration import _BetaCal, _BetaAMCal, _BetaABCal
from sklearn.base import BaseEstimator, RegressorMixin


class BetaCalibration(BaseEstimator, RegressorMixin):
    def __init__(self, parameters="abm"):
        if parameters == "abm":
            self.calibrator = _BetaCal()
        elif parameters == "am":
            self.calibrator = _BetaAMCal()
        elif parameters == "ab":
            self.calibrator = _BetaABCal()
        else:
            raise ValueError('Unknown parameters', parameters)

    def fit(self, X, y, sample_weight=None):
        self.calibrator.fit(X, y, sample_weight)
        return self

    def predict(self, S):
        return self.calibrator.predict(S)
