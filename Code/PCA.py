import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
data = pd.read_csv("/Users/watson/Downloads/data.csv",header=0)
data.drop("Unnamed: 32",axis=1,inplace=True)
data.drop("id",axis=1,inplace=True)
#features list of mean
features_mean= list(data.columns[1:11])
#features list of se
features_se= list(data.columns[11:21])
#features list of worst
features_worst=list(data.columns[21:31])
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
#split the data into train data and test data
train, test = train_test_split(data, test_size = 0.3)
#get Data of input dataSet and labelSet
def getData(var):
    train_X = train[var]
    train_Y = train.diagnosis
    test_X = test[var]
    test_Y = test.diagnosis
    return train_X,train_Y,test_X,test_Y
#get data after dimensional reduction
def pca(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    TwoD_Data = pca.transform(data)
    PCA_df = pd.DataFrame()
    PCA_df['PCA_1'] = TwoD_Data[:, 0]
    PCA_df['PCA_2'] = TwoD_Data[:, 1]
    return PCA_df
#cal the accuracy using SVM using the two-dimensional data
def svmWithPCA(var):
    train_X, train_Y, test_X, test_Y = getData(var)
    trainforSVM = pca(train_X)
    testforSVM = pca(test_X)
    model = svm.SVC(kernel="linear")
    model.fit(trainforSVM, train_Y)
    prediction = model.predict(testforSVM)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    return accuracy
print "Accuracy of SVM model(with linear kernel) after using pca built with original dataset:"
print "mean features:",svmWithPCA(features_mean)
print "standard error features:",svmWithPCA(features_se)
print "worst features:",svmWithPCA(features_worst)
print "------------------------------------------------------\n"
#plot pca
def drawPCA(var,labelx,labely):
    subdata=data[var]
    pca_data = pca(subdata)
    plt.figure(figsize=(14, 14))
    model = svm.SVC(kernel="linear")
    model.fit(pca_data, data.diagnosis)
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(pca_data['PCA_1']), max(pca_data['PCA_1']), 100)
    yy = a * xx - (model.intercept_[0]) / w[1]
    plt.plot(pca_data['PCA_1'][data.diagnosis == 1], pca_data['PCA_2'][data.diagnosis == 1], 'o', color="r",alpha=0.8)
    plt.plot(pca_data['PCA_1'][data.diagnosis == 0], pca_data['PCA_2'][data.diagnosis == 0], 'o', color='g', alpha=0.8)
    plt.plot(xx,yy)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.legend(['Malignant','Benign','Decision Line'])
    plt.show()
drawPCA(features_mean,"mean_1","mean_2")
drawPCA(features_se,"se_1","se_2")
drawPCA(features_worst,"worst_1","worst_2")