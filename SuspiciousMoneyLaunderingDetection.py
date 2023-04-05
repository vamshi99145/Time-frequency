from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix


main = tkinter.Tk()
main.title("A Time-Frequency Based Suspicious Activity Detection for Anti-Money Laundering") #designing main screen
main.geometry("1000x650")

global filename
global fscore
global X_train, X_test, y_train, y_test
global X, Y
global le1, le2, le3, dataset, rf
   
def upload():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('Label').size()
    label.plot(kind="bar")
    plt.title("Transaction Graph 0 Means Normal Transaction & 1 Means Money Laundering Transaction") 
    plt.show()

    
def preprocessDataset():
    global X, Y
    global le1, le2, le3, dataset
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    cols = ['type','nameOrig','nameDest']
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    dataset[cols[0]] = pd.Series(le1.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le2.fit_transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le3.fit_transform(dataset[cols[2]].astype(str)))

    Y = dataset['Label'].ravel()
    dataset.drop(['Label'], axis = 1,inplace=True)
    X = dataset.values
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Processed Dataset values\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset : "+str(X.shape[1] + 1)+"\n")
    
def runTransactionRandomForest():
    global X, Y, fscore, rf
    fscore = []
    text.delete('1.0', END)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    cls = RandomForestClassifier(ccp_alpha=0.5)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    f = f1_score(y_test, predict,average='macro') * 100
    fscore.append(f)
    CM = confusion_matrix(y_test, predict)
    TN = CM[0][0] / len(y_test)
    FN = CM[1][0] / len(y_test)
    TP = CM[1][1] / len(y_test)
    FP = CM[0][1] / len(y_test)
    cls = RandomForestClassifier()
    cls.fit(X_train, y_train)
    rf = cls
    text.insert(END,"Random Forest FSCORE on Transaction Data : "+str(f)+"\n")
    text.insert(END,"False Positive Rate (FPR) : "+str(FP)+"%\n")
    text.insert(END,"False Negative Rate (FNR) : "+str(FN)+"%\n")
    text.insert(END,"True Negative Rate (PPV) : "+str(TN)+"%\n")
    text.insert(END,"True Positive Rate (TPR) : "+str(TP)+"%\n\n")


def runTimeFrequencyRandomForest():
    global X, Y, fscore, rf
    fft_data = fft(X)
    X = []
    for i in range(len(fft_data)):
        temp = []
        for j in range(len(fft_data[i])):
            temp.append(float(fft_data[i,j]))
        X.append(temp)
    X = np.asarray(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    cls = RandomForestClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    f = f1_score(y_test, predict,average='macro') * 100
    fscore.append(f)
    CM = confusion_matrix(y_test, predict)
    TN = CM[0][0] / len(y_test)
    FN = CM[1][0] / len(y_test)
    TP = CM[1][1] / len(y_test)
    FP = CM[0][1] / len(y_test)
    text.insert(END,"Random Forest FSCORE on Transaction & Time Frequency Data : "+str(f)+"\n")
    text.insert(END,"False Positive Rate (FPR) : "+str(FP)+"%\n")
    text.insert(END,"False Negative Rate (FNR) : "+str(FN)+"%\n")
    text.insert(END,"True Negative Rate (PPV) : "+str(TN)+"%\n")
    text.insert(END,"True Positive Rate (TPR) : "+str(TP)+"%\n\n")
    

def predict():
    global rf, le1, le2, le3
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    cols = ['type','nameOrig','nameDest']
    dataset[cols[0]] = pd.Series(le1.transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le2.transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le3.transform(dataset[cols[2]].astype(str)))
    dataset = dataset.values
    predict = rf.predict(dataset)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"Transaction data : "+str(dataset[i])+" ====> PREDICTED AS NORMAL\n\n")
        if predict[i] == 1:
            text.insert(END,"Transaction data : "+str(dataset[i])+" ====> PREDICTED AS MONEY-LAUNDERING\n\n")    

def graph():
  height = fscore
  bars = ('Transaction Data FScore','Time Frequency Data FScore')
  y_pos = np.arange(len(bars))
  plt.bar(y_pos, height)
  plt.title("Random Forest FScore on Transaction & Time Frequency Data")
  plt.xticks(y_pos, bars)
  plt.show()

def close():
  main.destroy()
   
font = ('times', 16, 'bold')
title = Label(main, text='A Time-Frequency Based Suspicious Activity Detection for Anti-Money Laundering', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Anti-Money Laundering Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=10,y=150)
processButton.config(font=font1)

transactionButton = Button(main, text="Run Random Forest with Transaction", command=runTransactionRandomForest)
transactionButton.place(x=10,y=200)
transactionButton.config(font=font1)

tfButton = Button(main, text="Run With Transaction & Time Frequency", command=runTimeFrequencyRandomForest)
tfButton.place(x=10,y=250)
tfButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=10,y=300)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Money Laundering from Test Data", command=predict)
predictButton.place(x=10,y=350)
predictButton.config(font=font1)

closeButton = Button(main, text="Close Application", command=close)
closeButton.place(x=10,y=400)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
