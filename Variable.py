## 存储程序的变量


## 固定每个样本（视频）的 高、宽、深度（帧数),
## 读取视频前 多少秒 的图片帧
## 视频、数据 相对存储路径
#------------------------------------------------------------------------------------------
IMG_ROWS = 144
IMG_COLS = 176
IMG_DEPTH = 30
SECOND = 2
## 训练批次
EPOCH = 50
BATCH_SIZE = 8
ROOT = './DataSet/'

## 指令类别         
UAV = ['UAV', '前进', '后退', '左转', '右转', '上升', '下降', 
        '悬停', '降落', '加速', '减速', '向左', '向右' ]

Combat = ['Combat', '进攻', '防守', '撤退', '掩护我', '跟着我', '全体集合', '汇报情况', '坚持到底', 
            '占据要点', '此地安全', '发现敌人', '请求支援','收到','检查弹药', '隐蔽']

Action = ['Action', 'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
#---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
## 指令系统 类别(UAV, Combat, Action)
## 此变量，只有在切换 指令系统时需要改变
INSTRUCTION = Combat

## 根据人的不同，分为不同的数据源
## 1-50对应'p1/'；51-100对应'p2/';以此类推
## 此变量，在运行    collectData.py          ,且切换 不同人的数据 时需要改变
##        在运行    Instruction_train.py    时，选择 不同的人 的数据集 进行训练
##        在运行    test.py                  时，选择 不同的人 的数据集 进行测试
people = 'p1'
## 对应 原始avi文件的 文件夹名
## people 1 对应的是 1-50；people 2 对应的是 51-100......
## 此变量，在运行     collectData.py，          对同一个人，换不同批次数据时改变 
num_avi = 5

## 数据的模式，训练集（0）、测试集（1）
## 此变量，只有运行     collectData.py          才需要改变（以上选择后，对数据用途 的选择）
mode = 1
## 模型的路径
## 此变量，在运行       Instruction_train.py    ,选择在现有模型基础上迭代
##        在运行       test.py                  ,选择 预测的模型
model = '1.h5'
# ---------------------------------------------------------------------


## 视频存储路径
Instruciton_avi = ''

## mat数据(训练、验证)存储路径
Instruction_data = ''
## mat数据（测试集）存储路径
Instruction_data_test = '' 
## 根据不同的模式，数据存储路径不同（Instruction_data，Instruction_data_test）
mat_data = ''

UAV_data = ROOT + 'UAV/' + people + '/'
UAV_data_test = ROOT + 'UAV_test/' + people + '/'
UAV_avi = ROOT + 'UAV_avi/' + str(num_avi) 

Combat_data = ROOT + 'Combat/' + people + '/'
Combat_data_test = ROOT + 'Combat_test/' + people + '/'
Combat_avi = ROOT + 'combat_avi/' + str(num_avi) 

Action_data = ROOT + 'Action/' + people + '/'
Action_data_test = ROOT + 'Action_test/' + people + '/'
ActionRecognize_avi = ROOT + 'actionRecognize_avi/' + str(num_avi) 

if (INSTRUCTION == UAV):
    Instruciton_avi = UAV_avi
    Instruction_data = UAV_data
    Instruction_data_test = UAV_data_test 
elif(INSTRUCTION == Combat):
    Instruciton_avi = Combat_avi
    Instruction_data = Combat_data
    Instruction_data_test = Combat_data_test 
elif(INSTRUCTION == Action):
    Instruciton_avi = ActionRecognize_avi
    Instruction_data = Action_data
    Instruction_data_test = Action_data_test 

if(mode == 0):
    mat_data = Instruction_data
else:
    mat_data = Instruction_data_test


## 保存的.mat文件的形式
MAT = ['data_set', 'label_set', 'category']
