'''
    2019.09.16  15:20   ljp
    功能：
        将数据划分为 训练集 和 验证集，带入模型进行训练
        保存得到的模型参数，并显示在 训练集、验证集上的 准确率

    目的：
        熟悉python函数，plt画图

'''


## 定义、加载模型
from keras.models import load_model
## 实现one-hot格式（转换为one-hot）
from keras.utils import np_utils
## 数据集划分为训练集、测试集
from sklearn.model_selection import  train_test_split
## 画图
import matplotlib.pyplot as plt 
## 获取文件路径、生成文件等
import os
import numpy as np
## 回滚，动态改变学习率
from keras.callbacks import ReduceLROnPlateau

import Variable
import getData as gdata
import models


## 数据库 存储位置
PATH = Variable.Instruction_data
## 指令的类别数量（用于最后的输出层）
Num_CLASSES = len(Variable.INSTRUCTION) - 1
## 
EPOCH = Variable.EPOCH
BATCH_SIZE = Variable.BATCH_SIZE

IMG_ROWS = Variable.IMG_ROWS
IMG_COLS = Variable.IMG_COLS
IMG_DEPTH = Variable.IMG_DEPTH

X, Y = gdata.getData(PATH)

## ----------------------------------------------------------------------------------------

# X = X[:5, :, :, :, :]
# Y = Y[:5, :]

## ----------------------------------------------------------------------------------------

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

# write results to file
import datetime

## 获取当前时间
now = str(datetime.datetime.now()).split('.')[0]

filename = './results/' + now
time = now.replace(' ','_').replace(':','-')
storename = filename.replace(' ','_').replace(':','-')
filename = storename + '_' + str(EPOCH) + 'epoch' + '.txt'
people = Variable.people

def train():
    modleUsed = Variable.model
    model_exists = os.path.exists(modleUsed)
    if (model_exists):
        model = load_model(modleUsed)
        print("**************************************************")
        print(modleUsed," model loaded")

    else:
        model = models.model(IMG_COLS, IMG_ROWS, IMG_DEPTH, Num_CLASSES, 1)
        model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])

    # 划分数据集
    # 测试集所占比例为0.2(test_size)
    x_train, x_val, y_train,y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=4)

    model.summary()

    
    from keras.utils import plot_model
    # from keras.utils.vis_utils import plot_model
    from keras.utils.vis_utils import model_to_dot
    from IPython.display import SVG, Image
    # %matplotlib inline

    ## pip install GraphViz
    # 方法1;将网络格式存为图片格式
    plot_model(model, to_file= storename +'_model.png', show_shapes=True, show_layer_names=True, rankdir='LB')
    Image(storename +'_model.png')
    

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    import time as t
    t_star = t.time()
    # Train the model
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val,y_val),
        batch_size=BATCH_SIZE,
        # nb_epoch = EPOCH, ## keras新版本更改名字
        epochs= EPOCH,
        callbacks=[reduce_lr],
        shuffle=True
    )
    t_end = t.time()

    model.summary()


    #hist = model.fit(train_set, Y_train, batch_size=batch_size,
    #         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
    #           shuffle=True)


    # Evaluate the model
    score = model.evaluate(
        x_val,
        y_val,
        batch_size=BATCH_SIZE,
        #show_accuracy=True
        )

    # Save model
    model.save("./model_{}_{:.2f}_{}.h5".format(people,score[2], time))

    # Plot the results
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    print('\n***********************************************************************\n')
    print("训练用时：", t_end - t_star,"s")
    print('Test score:', score)
    print()
    # print('History', hist.history)
    # print()
    # print('train_loss', train_loss)
    # print()
    # print('val_loss', val_loss)
    # print()
    # print('train_acc', train_acc)
    # print()
    # print('val_acc', val_acc)




    # isExists = os.path.exists(filename)
    # if not isExists:
    #     os.makedirs(filename)
    ## 将训练的结果写入filename文件中
    target = open(filename, 'w')
    target.write('train_loss\n')
    target.write(str(train_loss))
    target.write('\nval_loss\n')
    target.write(str(val_loss))
    target.write('\ntrain_acc\n')
    target.write(str(train_acc))
    target.write('\nval_acc\n')
    target.write(str(val_acc))
    target.close()

    ## Epoch
    xc=range(50)

    ## 新建一个7*5 英寸的图形。图形id为1
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    ## 设置网格
    plt.grid(True)
    plt.legend(['train','val'])
    ## 图片美化，风格的选项如下
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['ggplot'])
    plt.savefig(storename +'_'+people+'_loss.png')

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    ## 设置网格
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig(storename +'_'+people+'_acc.png')
    plt.show()


if __name__ == "__main__":
    
    train()


'''
    结果：

    总结：
        ETA：Estimated Time of Arrival
        'ctrl' + '-' 和 'ctrl' + '+' 可以控制代码字体变小、变大
'''