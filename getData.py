'''
    2019.09.16  10:03   ljp
    功能：
        读取指定路径 中的.mat文件，将其值输出返回
        mat格式内容为 
            train_set：5-D double（（样本数，1，高，宽，帧数））
            label    ：1*num_samples int32
            category ：12*2 char（12个命令，每个命令2个char表示）

    目的：
        读取数据库数据进行操作，熟悉python存、读文件的操作

'''

## 获取文件路径、生成文件等
import os
## 保存数据成mat等格式---生成数据库；读取数据
import scipy.io as sio
import numpy as np

## 导入本地py文件（批量存储变量）
import Variable


mat = []
mat.append(Variable.MAT[0])
mat.append(Variable.MAT[1])


def getData(path):
    x, y = [], []
    emg_file = os.listdir(path)
    for emg in emg_file:
        emg_data = sio.loadmat(path + emg)
        data_x = emg_data[mat[0]]
        label_y = emg_data[mat[1]]
        x.append(data_x)
        y.append(label_y)
    X = np.concatenate(x, axis=0)
    Y = np.concatenate(y, axis=0)
    print('数据样本shape：', X.shape)
    print('数据标签shape：', Y.shape)
    
    return X, Y


'''
    结果：
        目录下所有mat格式的文件，提取出来为数组，返回给X、Y

    总结：

'''