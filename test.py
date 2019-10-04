import numpy as np 
from keras.models import load_model
## 实现one-hot格式（转换为one-hot）
from keras.utils import np_utils

import Variable
import getData as gdata

modelUsed = Variable.model
model = load_model(modelUsed)
Num_CLASSES = len(Variable.INSTRUCTION) - 1
BATCH_SIZE = Variable.BATCH_SIZE

PATH = Variable.Instruction_data_test
X, Y = gdata.getData(PATH)

#pred = model.predict(X)
#y_hat = np.argmax(pred)

## 将多余的维度（维度数量为1）去除
# Y_data = np.squeeze(Y)
## 将原始 标签 数据，转换为 one-hot编码
## 这个函数 不要求 输入数据（Y）一定是一维数组
Y_data = np_utils.to_categorical(Y, Num_CLASSES)

## 预处理，标准化
X_data = X.astype('float32')
X_data -= np.mean(X_data)
X_data /= np.max(X_data)
## 轴滚动：（样本数，1，高，宽，帧数）-->（样本数，高，宽，帧数，1）
X_data = np.rollaxis(X_data, 1, 5)
pred = model.predict(X_data)
y_hat = np.argmax(pred)
# Evaluate the model
score = model.evaluate(
    X_data,
    Y_data,
    batch_size=BATCH_SIZE,
    )
print(score)