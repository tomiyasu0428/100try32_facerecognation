import unittest
import numpy as np
from src.face_recognition import compare_vectors


class TestFaceRecognition(unittest.TestCase):
    def test_compare_vectors(self):
        # 同一ベクトルならスコアが1.0に近いはず
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        score = compare_vectors(vec1, vec2)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_compare_vectors_different(self):
        # 全く異なるベクトルならスコアが低い
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        score = compare_vectors(vec1, vec2)
        self.assertAlmostEqual(score, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
