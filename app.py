import os
import uuid
import json
from datetime import timedelta, datetime
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, request, redirect, url_for, flash, session
from PIL import Image, ImageDraw, ImageFont
import time

# =========================================
# Flaskアプリ設定
# =========================================
app = Flask(__name__)
app.secret_key = 'your-secret-key-keep-it-secret'  # 固定のシークレットキー
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 最大16MB
app.permanent_session_lifetime = timedelta(minutes=30)  # セッション有効期限を30分に延長

# =========================================
# 定数
# =========================================
FACE_RECOGNITION_THRESHOLD = 0.5  # 顔認識の類似度しきい値
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "registered_faces.json")
TEMP_DIR = os.path.join(BASE_DIR, "temp")  # 一時ファイル用ディレクトリ

# データディレクトリと一時ディレクトリが存在しない場合は作成
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# =========================================
# テンプレートフィルター
# =========================================
@app.template_filter('datetime')
def format_datetime(value):
    """Unix timestampを日時文字列に変換"""
    return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')

# =========================================
# 関数定義
# =========================================
def load_registered_faces():
    """登録済みの顔データを読み込む"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # データ形式の検証と移行
            needs_migration = False
            for features in data.values():
                if len(features) != 128:  # face_recognitionは128次元
                    needs_migration = True
                    break
            
            if needs_migration:
                # 古いデータを削除し、新しいデータベースを作成
                app.logger.warning("古い形式のデータベースを検出。データベースをリセットします。")
                data = {}
                save_registered_faces(data)
            
            return data
    return {}

def save_registered_faces(data):
    """顔データを保存する"""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def detect_faces(img):
    """
    画像から顔を検出する
    返り値: 顔の位置情報のリスト [(top, right, bottom, left), ...]
    """
    # 画像が大きすぎる場合はリサイズ（処理速度と精度のバランス）
    height, width = img.shape[:2]
    max_size = 1024
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size)
    
    # BGR -> RGB変換（face_recognitionはRGB形式を期待）
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        # まずHOGで試行（より高速）
        face_locations = face_recognition.face_locations(rgb_img, model="hog")
        
        # 顔が検出されない、または不自然な結果の場合はCNNで再試行
        if not face_locations or len(face_locations) > 10:  # 10人以上検出は不自然と判断
            face_locations = face_recognition.face_locations(rgb_img, model="cnn")
    except Exception as e:
        app.logger.error(f"顔検出エラー: {str(e)}")
        return []
    
    # 元のサイズに位置情報を戻す
    if height > max_size or width > max_size:
        scale = max(height, width) / max_size
        face_locations = [
            (int(top * scale), int(right * scale), 
             int(bottom * scale), int(left * scale))
            for top, right, bottom, left in face_locations
        ]
    
    return face_locations

def get_feature_vector(face_img):
    """
    顔画像から特徴ベクトルを抽出する
    返り値: 128次元の特徴ベクトル
    """
    try:
        # 顔画像のサイズを標準化（150x150ピクセル）
        face_img = cv2.resize(face_img, (150, 150))
        
        # BGR -> RGB変換
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # 顔の特徴を抽出（128次元ベクトル）
        encodings = face_recognition.face_encodings(rgb_img, num_jitters=3)
        
        if encodings:
            # 特徴量の次元数を確認
            if len(encodings[0]) != 128:
                app.logger.error(f"異常な特徴量の次元数: {len(encodings[0])}")
                return None
            return encodings[0]
        
        return None
    except Exception as e:
        app.logger.error(f"特徴抽出エラー: {str(e)}")
        return None

def find_closest_match(features):
    """
    特徴ベクトルに最も近い登録済みの顔を探す
    返り値: (ラベル, 類似度) または (None, None)
    """
    try:
        registered_faces = load_registered_faces()
        
        if not registered_faces:
            return None, None
        
        best_match = None
        best_similarity = -1
        
        for label, stored_features in registered_faces.items():
            # 特徴量の次元数を確認
            if len(stored_features) != 128:
                app.logger.error(f"保存された特徴量の次元数が不正: {len(stored_features)}")
                continue
            
            # face_recognitionの比較関数を使用
            distances = face_recognition.face_distance([np.array(stored_features)], features)
            similarity = 1 - distances[0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = label
        
        if best_similarity > FACE_RECOGNITION_THRESHOLD:
            return best_match, best_similarity
        return None, None
    except Exception as e:
        app.logger.error(f"顔認識エラー: {str(e)}")
        return None, None

def register_face(label, features):
    """新しい顔を登録する"""
    registered_faces = load_registered_faces()
    
    # 同じラベルが存在する場合は特徴量を平均化
    if label in registered_faces:
        stored_features = np.array(registered_faces[label])
        # 新しい特徴量と既存の特徴量を平均化
        features = (stored_features + features) / 2
    
    registered_faces[label] = features.tolist()
    save_registered_faces(registered_faces)

def draw_faces_with_labels(img, faces_info):
    """
    画像に顔のバウンディングボックスとラベルを描画
    faces_info: [(top, right, bottom, left, label, confidence), ...]
    """
    img_with_faces = img.copy()

    for i, (top, right, bottom, left, label, confidence) in enumerate(faces_info):
        # バウンディングボックスを描画
        cv2.rectangle(img_with_faces, (left, top), (right, bottom), (0, 255, 0), 2)

        # ラベルとconfidenceを表示
        if label:
            confidence_text = f"{int(confidence * 100)}%" if confidence else ""
            label_text = f"{label} {confidence_text}"
        else:
            label_text = f"未登録 #{i+1}"

        # 背景付きのテキスト表示（日本語対応）
        font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"  # macOSの場合
        font_size = 32
        img_pil = Image.fromarray(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)

        # テキストサイズを取得
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # 背景を描画
        draw.rectangle([(left, top - text_h - 10), (left + text_w + 10, top)], fill=(0, 255, 0))
        # テキストを描画
        draw.text((left + 5, top - text_h - 5), label_text, font=font, fill=(0, 0, 0))

        # PIL画像をOpenCV形式に戻す
        img_with_faces = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_with_faces

def save_temp_data(data):
    """一時データをファイルに保存"""
    temp_id = str(uuid.uuid4())
    temp_file = os.path.join(TEMP_DIR, f"{temp_id}.json")
    
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return temp_id
    except Exception as e:
        app.logger.error(f"一時ファイルの保存に失敗: {str(e)}")
        return None

def load_temp_data(temp_id):
    """一時データをファイルから読み込み"""
    if not temp_id:
        return None
        
    temp_file = os.path.join(TEMP_DIR, f"{temp_id}.json")
    try:
        if os.path.exists(temp_file):
            with open(temp_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        app.logger.error(f"一時ファイルの読み込みに失敗: {str(e)}")
    return None

def cleanup_temp_files(max_age_minutes=30):
    """古い一時ファイルを削除"""
    try:
        current_time = time.time()
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(filepath):
                file_age_minutes = (current_time - os.path.getmtime(filepath)) / 60
                if file_age_minutes > max_age_minutes:
                    os.remove(filepath)
                    app.logger.info(f"古い一時ファイルを削除: {filename}")
    except Exception as e:
        app.logger.error(f"一時ファイルのクリーンアップに失敗: {str(e)}")

# =========================================
# Flaskルーティング
# =========================================
@app.route("/", methods=["GET", "POST"])
def index():
    cleanup_temp_files()  # 古い一時ファイルを削除
    
    if request.method == "POST":
        if "file" not in request.files:
            flash("ファイルがアップロードされていません。")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("ファイルが選択されていません。")
            return redirect(request.url)

        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            flash("PNG、JPG、JPEG形式の画像ファイルを選択してください。")
            return redirect(request.url)

        # 画像を読み込んで顔検出
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        faces = detect_faces(img)
        if not faces:
            flash("画像から顔を検出できませんでした。")
            return redirect(request.url)

        faces_info = []
        unknown_faces = []
        
        for i, (top, right, bottom, left) in enumerate(faces):
            face_img = img[top:bottom, left:right]
            features = get_feature_vector(face_img)
            
            if features is not None:
                best_label, similarity = find_closest_match(features)
                if best_label and similarity > FACE_RECOGNITION_THRESHOLD:
                    faces_info.append((top, right, bottom, left, best_label, similarity))
                else:
                    faces_info.append((top, right, bottom, left, None, None))
                    unknown_faces.append({
                        "id": i + 1,
                        "coords": [top, right, bottom, left],
                        "features": features.tolist()
                    })

        # 顔を描画した画像を保存
        img_with_faces = draw_faces_with_labels(img, faces_info)
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        cv2.imwrite(save_path, img_with_faces)

        if unknown_faces:
            # 未認識の顔の情報を一時ファイルに保存
            temp_id = save_temp_data({
                "unknown_faces": unknown_faces,
                "filename": filename
            })
            
            flash(f"{len(faces)}人の顔を検出しました")
            flash(f"{len(unknown_faces)}人の未登録の顔があります")
            return redirect(url_for("label_faces", temp_id=temp_id))
        else:
            recognized_count = len([face for face in faces_info if face[4] is not None])
            flash(f"すべての顔({recognized_count}人)を認識しました！")
            return redirect(url_for("result", filename=filename))

    return render_template("index.html")

@app.route("/label_faces", methods=["GET", "POST"])
def label_faces():
    temp_id = request.args.get("temp_id")
    if not temp_id:
        app.logger.error("temp_idが見つかりません")
        flash("セッション情報が失われました。最初からやり直してください。")
        return redirect(url_for("index"))

    temp_data = load_temp_data(temp_id)
    if not temp_data:
        app.logger.error(f"一時データが見つかりません: temp_id={temp_id}")
        flash("セッション情報が失われました。最初からやり直してください。")
        return redirect(url_for("index"))

    if request.method == "POST":
        try:
            app.logger.info("ラベル登録処理開始")
            labels = request.form.getlist("labels[]")
            unknown_faces = temp_data.get("unknown_faces", [])
            
            app.logger.info(f"受け取ったラベル: {labels}")
            app.logger.info(f"未登録の顔の数: {len(unknown_faces)}")
            
            if len(labels) != len(unknown_faces):
                app.logger.error(f"ラベル数と顔の数が一致しません: labels={len(labels)}, faces={len(unknown_faces)}")
                flash("ラベル情報が不正です。")
                return redirect(url_for("index"))

            # 各顔を登録
            registered_count = 0
            for label, face_info in zip(labels, unknown_faces):
                if label.strip():  # ラベルが入力されている場合のみ登録
                    try:
                        features = np.array(face_info["features"])
                        register_face(label.strip(), features)
                        registered_count += 1
                        app.logger.info(f"顔を登録しました: {label}")
                    except Exception as e:
                        app.logger.error(f"顔の登録に失敗: {label}, エラー: {str(e)}")

            if registered_count > 0:
                flash(f"{registered_count}人の新規登録が完了しました！")
            else:
                flash("登録する顔が選択されていません。")
                
            # 一時ファイルを削除
            temp_file = os.path.join(TEMP_DIR, f"{temp_id}.json")
            if os.path.exists(temp_file):
                os.remove(temp_file)
                app.logger.info(f"一時ファイルを削除: {temp_id}")
                
            return redirect(url_for("result", filename=temp_data["filename"]))

        except Exception as e:
            app.logger.error(f"登録エラー: {str(e)}")
            flash(f"登録中にエラーが発生しました: {str(e)}")
            return redirect(url_for("index"))

    return render_template("label_faces.html", 
                         unknown_faces=temp_data["unknown_faces"],
                         filename=temp_data["filename"])

@app.route("/registered_faces")
def registered_faces():
    """登録済みの顔データを表示するページ"""
    registered_faces = load_registered_faces()
    
    # 各人物の特徴量から簡単な統計情報を計算
    face_data = []
    for name, features in registered_faces.items():
        features_array = np.array(features)
        stats = {
            "name": name,
            "mean": float(np.mean(features_array)),
            "std": float(np.std(features_array)),
            "min": float(np.min(features_array)),
            "max": float(np.max(features_array)),
            "registration_time": os.path.getmtime(DATA_FILE)
        }
        face_data.append(stats)
    
    # 登録日時でソート
    face_data.sort(key=lambda x: x["registration_time"], reverse=True)
    
    return render_template("registered_faces.html", face_data=face_data)

@app.route("/result")
def result():
    filename = request.args.get("filename")
    
    if not filename:
        flash("画像情報が見つかりません。")
        return redirect(url_for("index"))
        
    return render_template("result.html", filename=filename)

# =========================================
# メイン実行部
# =========================================
if __name__ == "__main__":
    app.run(debug=True)
