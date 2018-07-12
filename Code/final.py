import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
data = pd.read_csv("/Users/watson/Downloads/data.csv",header=0)
data.info()
data.drop("Unnamed: 32",axis=1,inplace=True)
data.drop("id",axis=1,inplace=True)
# print data.columns
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:21])
features_worst=list(data.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)
# print data["diagnosis"]
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

# print data["diagnosis"]
# print data.describe()
data_matrix=data.as_matrix()
# sns.countplot(data['diagnosis'],label="Count")
# sns.plt.show()
def drawCorr(var,label):
    labellist = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points',
                 'symmetry', 'fractal dimension']
    corr = data[var].corr()  # .corr is used for find corelation
    plt.figure(figsize=(14, 14))
    sns.heatmap(corr, cbar=False, square=True, annot=True, fmt='.2f', annot_kws={'size': 15},
                xticklabels=labellist, yticklabels=labellist,
                cmap='YlGnBu', linewidths=.5)
    # plt.xlabel(label)
    # plt.ylabel(label)
    plt.title(label)
    plt.show()
drawCorr(features_mean,"Correlation of features_mean")
train, test = train_test_split(data, test_size = 0.3)
def getData(var):
    train_X = train[var]  # taking the training data input
    train_Y = train.diagnosis
    test_X = test[var]
    test_Y = test.diagnosis
    return train_X,train_Y,test_X,test_Y
def randomForest(var):
    train_X,train_Y,test_X,test_Y=getData(var)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)
    return accuracy

prediction_var=["radius_mean","texture_mean","smoothness_mean","compactness_mean","symmetry_mean","fractal_dimension_mean"]
#pca
def pca(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    TwoD_Data = pca.transform(data)
    PCA_df = pd.DataFrame()
    PCA_df['PCA_1'] = TwoD_Data[:, 0]
    PCA_df['PCA_2'] = TwoD_Data[:, 1]
    return PCA_df
# train_s.drop("diagnosis",axis=1,inplace=True)
# test_s.drop("diagnosis",axis=1,inplace=True)
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
    colors = [[0, 1, 0], [0, 0, 1]]
    data_colors = [colors[lbl] for lbl in data.diagnosis]
    pca_data = pca(subdata)
    print pca_data
    plt.figure(figsize=(14, 14))
    model = svm.SVC(kernel="linear")
    model.fit(pca_data, data.diagnosis)
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(pca_data['PCA_1']), max(pca_data['PCA_1']), 100)
    yy = a * xx - (model.intercept_[0]) / w[1]
    # plt.scatter(pca_data['PCA_1'], pca_data['PCA_2'], c=data_colors, alpha=0.5)
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
#svm
def c_svm(var):
    train_X, train_Y, test_X, test_Y = getData(var)
    model = svm.SVC(kernel="linear")
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy=metrics.accuracy_score(prediction, test_Y)
    return accuracy
#se
corr = data[features_se].corr() # .corr is used for find corelation
# plt.figure(figsize=(14,14))
# sns.heatmap(corr, cbar = False,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
#            xticklabels= labellist, yticklabels= labellist,
#            cmap= 'YlGnBu',linewidths=.5)
# # sns.plt.show()
prediction_var_se=['radius_se', 'texture_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se']

randomForest(prediction_var_se)
corr = data[features_worst].corr() # .corr is used for find corelation
print corr
# plt.figure(figsize=(14,14))
# sns.heatmap(corr, cbar = False,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
#            xticklabels= labellist, yticklabels=labellist,cmap="YlGnBu",linewidths=.5)
# sns.plt.show()
prediction_var_worst=['radius_worst', 'texture_worst','smoothness_worst', 'compactness_worst', 'symmetry_worst', 'fractal_dimension_worst']
print "mean_randomforest=",randomForest(prediction_var)
print "mean_svm=",c_svm(prediction_var)
print "se_randomforest=",randomForest(prediction_var_se)
print "se_svm=",c_svm(prediction_var_se)
print "worst_randomforest=",randomForest(prediction_var_worst)
print "worst_svm=",c_svm(prediction_var_worst)