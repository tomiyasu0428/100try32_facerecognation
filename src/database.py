import json
import os
import numpy as np
from config.config import DATABASE_PATH
from src.face_recognition import compare_vectors


def load_database():
    """
    JSONファイルから登録済みの顔特徴量データをロードする。
    戻り値は { person_id: [feature_vector], ... } の形式を想定。
    """
    if not os.path.exists(DATABASE_PATH):
        return {}
    with open(DATABASE_PATH, "r") as f:
        return json.load(f)


def save_database(db):
    """
    登録済みの顔特徴量データをJSONファイルに保存する。
    """
    with open(DATABASE_PATH, "w") as f:
        json.dump(db, f)


def register_face(person_id, feature_vector):
    """
    新規または既存のperson_idに対応する特徴量を登録する。
    """
    db = load_database()
    db[person_id] = feature_vector.tolist()
    save_database(db)


def find_closest_match(feature_vector, threshold=0.5):
    """
    データベース内の各人物の特徴量と比較し、
    閾値以上のスコアを得た最も類似度の高い人物を返す。
    """
    db = load_database()
    best_match = None
    best_score = -1

    for person_id, vec in db.items():
        stored_vec = np.array(vec)
        score = compare_vectors(feature_vector, stored_vec)
        if score > best_score:
            best_score = score
            best_match = person_id

    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score
