from keras.models import load_model

from Gui import Gui
import Variable


## 测试模型
modelUsed = Variable.model
model = load_model(modelUsed)
## 识别指令
category = Variable.INSTRUCTION

Gui(category)