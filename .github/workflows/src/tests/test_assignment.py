import numpy as np
import pytest
from src.assignment import LinearRegressionFromScratch

def test_grading_logic():
    model = LinearRegressionFromScratch(lr=0.01)
    model.initialize_weights(n_features=3)

    # 1. 차원 검증
    assert model.w.shape == (3, 1), "가중치 w의 차원이 (n_features, 1)이 아닙니다."

    # 2. 연산 결과 검증
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model.w = np.array([[1.0], [1.0], [1.0]])
    model.b = 1.0

    pred = model.forward(X)
    expected = np.array([[7.0], [16.0]])

    np.testing.assert_array_almost_equal(pred, expected, err_msg="Forward 연산 결과가 틀렸습니다.")
