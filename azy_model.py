import os
import gc
import numpy as np
from numpy.lib.npyio import save
from tensorflow import keras
from keras.engine.training import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPool2D,BatchNormalization, ReLU
from tensorflow.python.keras.layers.merge import add
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.models import load_model
from keras.callbacks import CSVLogger

# 各種パラメータ
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用するGPUのデバイス番号, GPUがないマシンで動かすときはコメントアウトしましょう
csv_logger = CSVLogger('model_history_azy.csv', append=True) # CSVでhsitoryを保存
loops = 232 # 暫定
# 画像
x_train_name = "trains/x_train_112.npy" # 訓練画像のパス
height = 112#224 # 入力画像の高さ
width = 112#224 # 入力画像の幅
channel = 3 # カラー画像
# ラベル
y_train_name = "trains/y_train_112.npy" # 小分類ラベルのパス
classes1_num = 70 # クラス数(小分類70個)
z_train_name = "trains/z_train_112.npy" # 中分類ラベルのパス
classes2_num = 14  # クラス数(中分類 14個)
a_train_name = "trains/a_train_112.npy" # 大分類ラベルのパス
classes3_num = 3  # クラス数(大分類 3個)


# 訓練データ読み込み
# 訓練画像読み込みと分割
x_train = np.load(x_train_name)
print(f'--- load image data from {x_train_name}')
print('shape :', x_train.shape)
x_train = x_train.reshape(-1, height, width, channel) # numpy配列の形状変換
print('reshape :', x_train.shape)
x_train_normalized = np.array_split(x_train, loops)
print('Split ' + x_train_name)
# 小分類ラベル読み込み
y_train = np.load(y_train_name)
print(f'--- load label data from {y_train_name}')
print('shape :', y_train.shape) # シェイプ確認
y_train = np.array_split(y_train, loops)
print('Split ' + y_train_name)
# 中分類ラベル読み込み
z_train = np.load(z_train_name)
print(f'--- load label data from {z_train_name}')
print('shape :', z_train.shape) # シェイプ確認
z_train = np.array_split(z_train, loops)
print('Split ' + z_train_name)
# 大分類ラベル読み込み
a_train = np.load(a_train_name)
print(f'--- load label data from {a_train_name}')
print('shape :', a_train.shape) # シェイプ確認
a_train = np.array_split(a_train, loops)
print('Split ' + a_train_name)
print('loops is '+str(loops))

# モデル構築
input_tensor = Input(shape=(height,width,channel))
add_layer = Conv2D(64, 7, padding='same', activation='relu', name='common_conv2d_1') (input_tensor)
add_layer = BatchNormalization() (add_layer)
add_layer = ReLU() (add_layer)
add_layer = Conv2D(64, 7, padding='same', activation='relu', name='common_conv2d_2') (add_layer)
add_layer = BatchNormalization() (add_layer)
add_layer = ReLU() (add_layer)
# add_layer = Conv2D(64, 3, padding='same', activation='relu', name='common_conv2d_3') (add_layer)
add_layer = MaxPool2D() (add_layer)
prediction3 = GlobalAveragePooling2D() (add_layer)
prediction3 = Dense(1024, activation='relu') (prediction3)
prediction3 = Dense(classes3_num, activation='softmax', name='a_label') (prediction3) # 大分類出力

add_layer = Conv2D(128, 5, padding='same', activation='relu', name='common_conv2d_4') (add_layer)
add_layer = BatchNormalization() (add_layer)
add_layer = ReLU() (add_layer)
add_layer = Conv2D(128, 5, padding='same', activation='relu', name='common_conv2d_5') (add_layer)
add_layer = BatchNormalization() (add_layer)
add_layer = ReLU() (add_layer)
add_layer = Conv2D(128, 5, padding='same', activation='relu', name='common_conv2d_6') (add_layer)
add_layer = BatchNormalization() (add_layer)
add_layer = ReLU() (add_layer)
add_layer = MaxPool2D() (add_layer)
prediction2 = GlobalAveragePooling2D() (add_layer)
prediction2 = Dense(1024, activation='relu') (prediction2)
prediction2 = Dense(classes2_num, activation='softmax', name='z_label') (prediction2) # 中分類出力

add_layer = Conv2D(256, 3, padding='same', activation='relu', name='common_conv2d_7') (add_layer)
add_layer = BatchNormalization() (add_layer)
add_layer = ReLU() (add_layer)
add_layer = Conv2D(256, 3, padding='same', activation='relu', name='common_conv2d_8') (add_layer)
add_layer = BatchNormalization() (add_layer)
add_layer = ReLU() (add_layer)
#add_layer = MaxPool2D() (add_layer)
add_layer = Conv2D(256, 3, padding='same', activation='relu', name='common_conv2d_9') (add_layer)
add_layer = BatchNormalization() (add_layer)
add_layer = ReLU() (add_layer)
#add_layer = Conv2D(512, 3, padding='same', activation='relu', name='common_conv2d_10') (add_layer)
prediction1 = GlobalAveragePooling2D() (add_layer)
prediction1 = Dense(1024, activation='relu') (prediction1)
prediction1 = Dense(classes1_num, activation='softmax', name='y_label') (prediction1) # 小分類出力

model = Model(inputs=input_tensor, outputs=[prediction3, prediction2, prediction1])

adam = keras.optimizers.Adam(lr=0.00001)
model.compile(
    optimizer=adam,
    loss={
        'y_label':'sparse_categorical_crossentropy',
        'z_label':'sparse_categorical_crossentropy',
        'a_label':'sparse_categorical_crossentropy'
    },
    metrics=['accuracy']
)
model.summary()
model.save('models/azy_112.h5')

for i in range(loops):
    # 学習(訓練)
    print('*****' + str(i) + '*****') # ループカウンタ(目視)
    number_of_epochs = 32 # エポック数(学習の回数)
    x_train_split = x_train_normalized[i].astype('float32') / 255.0 #分割されたx_train_nomalizedをfloat32に変換 計算の高速化を図る
    hist = model.fit(
        x_train_split, # 訓練画像
        {
            'y_label':y_train[i], 
            'z_label':z_train[i], 
            'a_label':a_train[i]
        },
        batch_size = 64, # バッチサイズ(一度の学習に使用するデータセットの数)
        epochs = number_of_epochs, # エポック数(学習の回数)
        callbacks = [csv_logger], # コールバック
        validation_split = 0.2 # データセットをバリデーションデータにする割合
        )
    model.save('models/azy_112.h5') #modelsディレクトリへ保存
    print(str(i) + " model saved.")
    gc.collect() # garbage collect

print('All process done.') # おわり

