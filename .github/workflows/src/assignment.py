import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.w = None
        self.b = None

    def initialize_weights(self, n_features):
        # TODO: 가중치 w를 (n_features, 1) 차원의 영행렬로, b를 0.0으로 초기화하세요.
        self.w = None
        self.b = None

    def forward(self, X):
        # TODO: 선형 회귀 예측 식을 NumPy로 구현하세요. (y = Xw + b)
        return None
