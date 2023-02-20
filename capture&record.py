# 必要なライブラリのインポート
import cv2
import datetime
import os

# Webカメラを使うときはこちら
cap = cv2.VideoCapture(0)

before = None
time = datetime.date.today()

# 保存先フォルダの作成
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# 保存する動画のファイル名
video_filename = "output_{}.mp4".format(
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)

# 動画保存用の設定
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    os.path.join(output_dir, video_filename), fourcc, 30.0, (640, 480)
)

print("動体検知開始")
print(str(datetime.datetime.now()))

while True:
    # 画像を取得
    ret, frame = cap.read()

    # 白黒画像に変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if before is None:
        before = gray.astype("float")
        continue

    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, before, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(before))

    # frameDeltaの画像を２値化
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]

    # 輪郭のデータを取得
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 差分があった点を画面に描画
    for target in contours:
        x, y, w, h = cv2.boundingRect(target)

        # 小さい変更点は無視
        if w < 300:
            continue

        # オブジェクトの位置に緑色の四角形を描画
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ウィンドウで表示
    cv2.imshow("target_frame", frame)

    # 動画を保存
    out.write(frame)

    # Enterキーが押されたらループを抜ける
    if cv2.waitKey(1) == 13:
        break

print("動体検知終了")
print(str(datetime.datetime.now()))

# 動画保存用の設定を解放
out.release()

# ウィンドウの破棄
cv2.destroyAllWindows()
