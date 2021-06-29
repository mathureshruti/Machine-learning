from tkinter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import tensorflow as tf
from joblib import dump, load
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
import sys

n_features = 3

class MyWindow:
    def __init__(self, win):
        self.lbl1=Label(win, text='Path for the file:')
        self.lbl1.place(x=100, y=50)
        self.t1=Entry(bd=3)
        self.t1.place(x=200, y=50)        
        
        self.lbl2=Label(win, text='Row number:')
        self.lbl2.place(x=80, y=100)
        self.t2=Entry(bd=3)
        self.t2.place(x=200, y=100)

        self.btn1 = Button(win, text='click', bg="green", command=self.predict)
        self.btn1.place(x=200, y=150)
        
        self.lbl3=Label(win, text='Result:',bg="yellow")
        self.lbl3.place(x=150, y=200)
        self.t3=Entry()
        self.t3.place(x=200, y=200)


    def predict(self):
        self.t3.delete(0, 'end')
        dataframe = pd.read_excel(str(self.t1.get()), header= None)
        data = np.array(dataframe)
        feature_array = []
        Normdata = np.zeros((data.shape[0],data.shape[1]))
        for i in range(data.shape[0]):
            Normdata[i] = data[i]/max(abs(data[i]))
            spectrum, freq, t, im = plot.specgram(Normdata[i], NFFT=256, Fs=8,  noverlap=0)
            feature_array.append([np.sum(spectrum),np.argmax(spectrum),np.where(spectrum == np.amax(spectrum))[1][0]])
            plot.xlabel('Time')
            plot.ylabel('Frequency')

        feature_array = np.array(feature_array)
        feature_array = feature_array.reshape(feature_array.shape[0],n_features)
        model1 = joblib.load('logistics Regression.joblib')
        y_pred = model1.predict(feature_array)
        if (y_pred[int(self.t2.get())] == 0):
            self.t3.insert(END, str('This is Object #1'))
        else:
            self.t3.insert(END, str('This is Not Object #1'))     



dataframe1 = pd.read_excel(r'D:\PROFILE BACKUP\Desktop\Final ML\Object 1\Obj1.xlsx', header=None)
Data1_array = np.array(dataframe1)

feature_array1 = []
Normdata1 = np.zeros((Data1_array.shape[0],Data1_array.shape[1]))
for i in range(Data1_array.shape[0]):
    Normdata1[i] = Data1_array[i]/max(abs(Data1_array[i]))
    spectrum, freq, t, im = plot.specgram(Normdata1[i], NFFT=256, Fs=8, noverlap=0)
    feature_array1.append([np.sum(spectrum),np.argmax(spectrum),np.where(spectrum == np.amax(spectrum))[1][0]])
    plot.close()
    

dataframe2 = pd.read_excel(r'D:\PROFILE BACKUP\Desktop\Final ML\Not Object 1\Notobj1.xlsx', header=None)
Data2_array = np.array(dataframe2)


feature_array2 = []
Normdata2 = np.zeros((Data2_array.shape[0],Data2_array.shape[1]))
for j in range(Data2_array.shape[0]):
    Normdata2[j] = Data2_array[j]/max(abs(Data2_array[j]))
    spectrum, freq, t, im = plot.specgram(Normdata2[j], NFFT=256, Fs=8, noverlap=0)
    feature_array2.append([np.sum(spectrum),np.argmax(spectrum),np.where(spectrum == np.amax(spectrum))[1][0]])
    plot.close()
    
# FEATURE DEFINING IN ARRAY FOR DATA1
feature_array1 = np.array(feature_array1)
feature_array1 = feature_array1.reshape(feature_array1.shape[0],n_features)

# FEATURE DEFINING IN ARRAY FOR DATA2
feature_array2 = np.array(feature_array2)
feature_array2 = feature_array2.reshape(feature_array2.shape[0],n_features)


X = np.vstack((feature_array1,feature_array2)) # Features
Y = np.hstack((np.zeros(feature_array1.shape[0]),np.ones(feature_array2.shape[0]))) # Labels


# Load and split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#Logistic Regression Model

model = LogisticRegression()

#Train The Model 
model.fit(x_train, y_train)

#Save the Model 
dump(model,'logistics Regression.joblib')

# Load from file
joblib_model = joblib.load('logistics Regression.joblib')

# Calculate the accuracy and predictions for the model
score = joblib_model.score(x_test, y_test)
print("Score: {0:.2f} %".format(100 * score))
y_predt = joblib_model.predict(x_test)


#Confusion Matrix
cm = confusion_matrix(y_test, y_predt).ravel()
tp, fp, fn, tn  = cm
disp = ConfusionMatrixDisplay(confusion_matrix=cm.reshape(2,2))
disp.plot()

print("True Positive:",tp)
print("False Positive:",fp)
print("False Negative:",fn)
print("True Negative:",tn)


#False Discovery Rate (FDR)
fdr = fp/(fp + tp)
print("False Discovery Rate (FDR): ",fdr)


#Negative Predictive Value (NPV)
npv = tn/(tn + fn)
print("Negative Predictive Value (NPV): ",npv)

#True Positives Rate (TPR)
tpr = tp/(tp + fn)
print("True Positives Rate (TPR): ",tpr)

#True Negatives  Rate (TNR)
tnr = tn/(tn + fp)
print("True Negatives  Rate (TNR): ",tnr)

#F1 score
F1 = 2*tp/((2*tp) + fp + fn)
print("F1 Score: ",F1)

#predict probabilities
probs = model.predict_proba(x_test)

#Keeping only positive class
probs = probs[:, 1]
 
#ROC
fpr1, tpr1, thresholds = roc_curve(y_test, probs)
 
#Plotting the ROC curve figure
plot.figure(figsize = (10,6))
plot.plot(fpr1, tpr1, color='blue', label='ROC')
plot.plot([0, 1], [0, 1], color='green', linestyle='--')
plot.xlabel('1-Specificity')
plot.ylabel('Sensitivity')
plot.title('Receiver Operating Characteristic Curve (ROC)')
plot.legend()
plot.show()

#Accuracy
Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
print("Accuracy {:0.2f}%:".format(Accuracy))


#Precision 
Precision = tp/(tp+fp) 
print("Precision {:0.2f}".format(Precision))

#ROC PRINT
print('roc_auc_score for Model (Logistic Regression): ', roc_auc_score(y_test, probs))

window=Tk()
mywin=MyWindow(window)
window.title('Discrimination of reflected sound signals')
window.geometry("500x400+10+10")
window.mainloop()