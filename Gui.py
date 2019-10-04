import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


## 用于管理部件上面的字符；不过一般用在按钮button上。改变StringVar，按钮上的文字也随之改变。
## 此处是用于 界面中上部的，label控件的提示信息
var1 = tk.StringVar()
label1 = tk.Label(window, textvariable=var1, width=20, height=1, font=('楷体', 50))

## 指令列表
var2 = tk.StringVar()
var2.set(category[1:])
lb = tk.Listbox(window, listvariable=var2, width=20, height=15, font=('楷体', 30))

## 识别结果提示，右上角
var3 = tk.StringVar()
label2 = tk.Label(window, textvariable=var3, width=20, height=1, font=('楷体', 50))

## 正确识别提示
var4 = tk.StringVar()
var4.set(0)
label6 = tk.Label(window, textvariable=var4, width=6, height=1, font=('楷体', 30))

## 错误识别提示
var5 = tk.StringVar()
var5.set(0)
label7 = tk.Label(window, textvariable=var5, width=6, height=1, font=('楷体', 30))

## 准确率显示
var6 = tk.StringVar()
label8 = tk.Label(window, textvariable=var6, width=8, height=1, font=('楷体', 30))

correct = 0
wrong = 0

def Gui(category, data_x):
    window = tk.Tk()
    window.title("TAIIC-缄默通信系统")
    ## w*h  +/-  x  +/-  y的格式;宽，高，往 宽方向 右移移动距离，往 高方向 下移距离
    window.geometry("1900x1010+0+0")

    q = tk.Button(window, text='Quit', width=6, height=1, command=window.quit, font=('Consolas', 25))
    ## 使用绝对坐标放置按钮，西北方向
    q.place(x=20, y=15, anchor='nw')

    label1.place(x=620, y=30)

    lb.place(x=50, y=150)

    ## 选中列表中的item，点击按钮，触发 command 命令
    button1 = tk.Button(window, text='选择', width=5, height=1, font=('楷体', 30), command=print_selection)
    button1.place(x=180, y=800)

    label2.place(x=620, y=320)


    click_me(data_x)

    button2 = tk.Button(window, text='Recode', width=8, height=1, command=click_me, font=('Consolas', 35))
    button2.pack(side='bottom')

    label3 = tk.Label(window, text='正确次数：', width=10, height=1, font=('楷体', 30))
    label3.place(x=1500, y=280)
    label4 = tk.Label(window, text='错误次数：', width=10, height=1, font=('楷体', 30))
    label4.place(x=1500, y=330)
    label5 = tk.Label(window, text='准确率：', width=10, height=1, font=('楷体', 30))
    label5.place(x=1500, y=380)

    label6.place(x=1660, y=280)

    label7.place(x=1660, y=330)

    label8.place(x=1660, y=380)

    window.mainloop()

## 将列表中选中的item显示出来
def print_selection():
    value = lb.get(lb.curselection())
    var1.set('所选指令: ' + value)


def click_me(data_x):
    global correct, wrong
    #--------------------------------------
    ## 得到数据
    data_x = data_x
    #------------------------------------------
    pred = model.predict(data_x)
    ## 返回最大概率的 下标
    y = np.argmax(pred)
    # print(pred)
    # print(category[y])
    result = category[y+1]
    # speaker = win32com.client.Dispatch("SAPI.SpVoice")
    # speaker.Speak(category[y])

    var3.set('识别结果: ' + result)
    if result == lb.get(lb.curselection()):
        correct += 1
        var4.set(correct)
        ## 弹窗提示
        messagebox.showinfo(title='识别结果',  message=' 正确 ! ')
    else:
        wrong += 1
        var5.set(wrong)
        messagebox.showerror(title='识别结果', message=' 错误 ! ')
    var6.set('{:^.2%}'.format(correct/(correct+wrong)))
