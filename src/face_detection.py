import cv2
import numpy as np
from insightface import model_zoo

# もし .env などで MODEL_PATH を指定している場合は、config.py から読み込む
# from config.config import MODEL_PATH

# ---------------------------------------------------
# ArcFaceモデルのロード（初回は自動的にモデルをダウンロードする場合あり）
# ---------------------------------------------------
arcface_model = model_zoo.get_model("arcface_r100_v1")
# CPUモード: ctx_id=-1, GPUモード: ctx_id=0 (GPUを認識していれば)
arcface_model.prepare(ctx_id=-1)


def get_feature_vector(face_img: np.ndarray) -> np.ndarray:
    """
    入力: 顔領域のRGB画像 (112x112推奨)
    出力: ArcFaceの埋め込みベクトル (512次元)
    """
    # InsightFaceのモデルはRGB形式を想定しているので、BGRの場合は変換
    # OpenCVで読み込んだ画像はBGRなので、RGBに変換
    if face_img.shape[2] == 3:  # カラー画像であることを確認
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # ArcFaceモデルでは、(112, 112)程度のサイズが推奨されている
    # 必要に応じてリサイズ
    face_img_resized = cv2.resize(face_img, (112, 112))

    # 埋め込みベクトルを取得
    embedding = arcface_model.get_embedding(face_img_resized)
    # 返り値は1 x 512のndarray
    # shapeを(512,)にして返す
    return embedding.flatten()


def compare_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    コサイン類似度を計算してスコアを返す。
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 * norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
