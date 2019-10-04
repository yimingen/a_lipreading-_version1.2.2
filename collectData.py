'''
    2019.09.15  22:03   ljp
    功能：
        读取某一任务指令的 视频文件夹中的 所有视频，转存为.mat格式
        mat格式内容为 
            train_set：5-D double（（样本数，1，高，宽，帧数））
            label    ：1*num_samples int32
            category ：12*2 char（12个命令，每个命令2个char表示）

    目的：
        实现视频流的数字化，熟悉python编程（各种函数的使用）

'''


## 获取文件路径、生成文件等
import os
import numpy as np 
import cv2
## 保存数据成mat等格式---生成数据库;读取数据
import scipy.io as sio

## 导入本地py文件（批量存储变量）
import Variable


## 切换不同任务时，注意改变 Variable 出现的地方

## 固定每个样本（视频）的 高、宽、深度（帧数),
## 读取视频前 多少秒 的图片帧
## 视频 相对存储路径
IMG_ROWS = Variable.IMG_ROWS
IMG_COLS = Variable.IMG_COLS
IMG_DEPTH = Variable.IMG_DEPTH
SECOND = Variable.SECOND
root_video = Variable.Instruciton_avi

## X_tr为列表，存储整个数据集(样本)
## Y_data存标签
X_data = []  
Y_data = []

## 指令类别         
Category = Variable.INSTRUCTION
## 数据的模式，训练集（0）、测试集（1）
mode = Variable.mode
## mat数据存放路径(根据mode 而不同)
path_save = ''

## mat数据存放路径
if(mode == 0):
    path_save = Variable.mat_data + 'semg' + Category[0] + '_train'
elif(mode == 1):
    path_save = Variable.mat_data + 'semg' + Category[0] + '_test'
## mat路径下，不同批次数据命名区别（与avi命名 保持一致）
name_save = Variable.num_avi


## 函数，存储数据为mat格式
# data_set ： data_set-----（样本数，1，高，宽，帧数）
# label_set ： label_set----（样本数，1）
# category ： category
# people：不同的人的数据，1代表ljp，2代表yh
# 【】代码修改后，通过文件夹名确定不同的人，people参数不用，但暂时保留
def save_data(data_set, label_set, category):
    data_sample = {}
    data_sample[Variable.MAT[0]] = data_set
    data_sample[Variable.MAT[1]] = label_set
    data_sample[Variable.MAT[2]] = category

    path = path_save

    ## 存储在 DataSet/[类别]/下；文件名为semg[类别]numSession.mat
    # numSession = 1
    # if(people == 2):
    #     numSession = 51
    # dataname = path + str(numSession) + '.mat'
    # while os.path.exists(dataname):
    #     numSession += 1
    #     dataname = path + str(numSession) + '.mat'
    dataname = path + str(name_save) + '.mat'
    while os.path.exists(dataname):
        print("该数据已存储！")
        return 

    ## mat格式存储data_sample,命名为dataname
    sio.savemat(dataname, data_sample)
    print("存储位置为: ", dataname)



if __name__ == "__main__":

    ## flag 作为指令的代号（从0开始）
    flag = -1

    ## 选取一套指令，读取对应指令集的目录文件（UAV，root_video1）---指令
    for i in range(1, len(Category)):
        instruction = Category[i]
        path_video = os.listdir(root_video + '/' + instruction)

        ## 应该 flag = i-1；此步多余 
        flag += 1
        ## 某一指令下，依次读取不同的样本---同一指令 不同样本
        for k in path_video:
            ## 视频的相对路径
            video = root_video + '/' + instruction + '/' + k
            frames = []
            cap = cv2.VideoCapture(video)
            ## 视频帧率（float型）
            # a = cap.get(5)
            fps = int(cap.get(5))
            ## SECOND为float型，与fps相乘后，为float型
            num_Frames = int(fps * SECOND)

            ## 读取每一帧---样本中每一帧
            for num in range(num_Frames):
                ## frame为一个（宽，高，通道）3维数组
                ret, frame = cap.read()
                ## 将frame 线性内插，统一化为（IMG_ROWS, IMG_COLS, 通道）的数组
                frame = cv2.resize(frame, (IMG_ROWS, IMG_COLS), interpolation=cv2.INTER_AREA)
                ## frame灰度转换，变为（宽，高）2维数组
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #---------------------------------------------------------------------------------------------------------
                ## frames为列表（list）; 读取 前30帧中的20帧，后30帧中的10帧
                if( ((num < 30)and(num % 3 != 0)) or ((num >= 30)and((num-1) % 3 == 0)) ):
                    frames.append(gray)
                #---------------------------------------------------------------------------------------------------------

                # ## frames为列表（list）; 
                # frames.append(gray)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            ## 将frames列表 数组化（帧数，宽， 高）
            array_frames = np.array(frames)
            ## 轴滚动：（帧数，宽， 高）-->（高，帧数，宽）-->（宽，高，帧数）
            # sample = np.rollaxis(np.rollaxis(array_frames, 2, 0), 2, 0)
            ## 轴滚动：（帧数，宽， 高）-->（宽， 高，帧数）
            sample = np.rollaxis(array_frames, 0, 3)
            ## label:(1,)的1维数组
            label = np.array([flag])

            ## X_data为 list列表
            X_data.append(sample)
            Y_data.append(label)
            
        
        # print(instruction + " done")
        # print("{:>6} done".format(instruction))
        # print("{:>6}\tdone".format(instruction))
        # print("{:^6}\tdone".format(instruction))
        print("{:<6}\tdone".format(instruction))
    print()

    ## 将 X_data列表 转换为数组 X_data_array---（样本数，宽， 高，帧数）
    X_data_array = np.array(X_data)
    ## 将 Y_data列表 转换为数组 Y_data_array---（样本数，1）
    Y_data_array = np.array(Y_data)

    num_samples = len(X_data_array)
    print("样本数量为：{:<4}个".format(num_samples))

    # ## data是一个list
    # ## data[0] = X_data_array--（样本数，宽， 高，帧数）
    # ## data[1] = Y_data_array--（样本数，1）
    # data = [X_data_array, Y_data_array]

    ## 数据集以（样本数，1，宽， 高，帧数）形式存储
    ## float64
    x_data_set = np.zeros((num_samples, 1, IMG_COLS, IMG_ROWS, IMG_DEPTH))
    # x_data_set = np.zeros((num_samples, 1, IMG_COLS, IMG_ROWS, IMG_DEPTH), dtype=int)

    for h in range(num_samples):
        x_data_set[h][0][:][:][:]=X_data_array[h,:,:,:]

    ## float64--->int16
    x_data_set = x_data_set.astype('int16')
    print("存储样本数据类型为：", x_data_set.dtype)

    # print('data_set: ', x_data_set)
    # print('labe: ', Y_data_array)

    save_data(x_data_set, Y_data_array, Category)


'''
    结果：
        视频信息存储在 DataSet/[类别]/下；文件名为semg[类别]numSession.mat

    总结：
        错误--SyntaxError: invalid character in identifier（标志符），　但又一直找不到问题点的话，
                请确保代码行内没有夹杂 中文输入法 的 空格， tab， 括号（‘（）’）等，非文字字符．
                *******这个错误很常见
        --numpy.AxisError: 'start' arg requires -1 <= start < 2, but 3 was passed in
        样本视频有问题，打不开
        --OverflowError: Python int too large to convert to C long;当一次处理的样本数过大时，sio.savemat处会出现这个错误
        astype('int16'),将np.zeros产生的默认float64类型数组，转换为int16，问题解决，原因不明
        
        提高代码的复用性，对不同的任务，函数主体使用相同的变量表示（可将变量 批量保存在一个文件中，此处是增加了Variable.py,
        更常用的方法是，添加main.py,变量用argparse添加）
'''





