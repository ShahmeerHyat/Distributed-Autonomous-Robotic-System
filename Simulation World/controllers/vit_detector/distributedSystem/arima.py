"""
arima.py  —  Copied from Edge/ARIMA.py (SPViT paper).

predict_next() is added on top of the original class so the adaptive
partition manager can get a single scalar forecast from a history list.
The paper's _ARIMA builds difference coefficients; predict_next uses the
mean of the last p differences as the AR forecast (standard AR(p,d=1) step).
"""

import numpy as np


class ARIMA:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q

    def difference(self, data, interval):
        return [data[i] - data[i - interval] for i in range(interval, len(data))]

    def inverse_difference(self, history, yhat, interval):
        return yhat + history[-interval]

    def fit(self, data):
        self.data = data
        self.history = [x for x in data]
        self.residuals = []
        return self

    def forecast(self):
        predictions = []
        for t in range(len(self.data)):
            model = self._ARIMA()
            yhat = model['coef'][0]
            obs = self.data[t]
            predictions.append(yhat)
            error = obs - yhat
            self.residuals.append(error)
        return predictions

    def _ARIMA(self):
        history = self.history
        model = {'coef': [0.0 for _ in range(self.p)]}
        for t in range(self.p, len(history)):
            model['coef'].append(history[t] - history[t - 1])
        return model

    # ── Added: single-step forecast used by PartitionManager ─────────────────
    def predict_next(self, history):
        """
        Predict the next value in `history` using AR(p, d=1).

        Steps:
          1. Difference the series (d=1): diffs[i] = history[i] - history[i-1]
          2. AR forecast: next_diff = mean of last min(p, len(diffs)) differences
          3. Invert: prediction = history[-1] + next_diff

        Falls back to the last observed value when there is not enough history.
        """
        if len(history) < 2:
            return history[-1] if history else 0.0

        diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
        window = diffs[-self.p:] if len(diffs) >= self.p else diffs
        next_diff = float(np.mean(window))
        return max(1e-9, history[-1] + next_diff)
