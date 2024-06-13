#!/usr/bin/env python
# coding: utf-8

# # 读取数据

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.dpi'] = 150 # 修改图片分辨率
plt.rcParams.update({'font.size': 12})
plt.rc('font',family='Times New Roman')


# In[5]:


excel_file = pd.ExcelFile('数据简化特征版本.xlsx')
sheet_names = excel_file.sheet_names
for sheet_name in sheet_names:
    print(sheet_name)


# In[2]:


y_list=['postprocedural_stroke']


# In[7]:


data=pd.read_excel('数据简化特征版本.xlsx',sheet_name='stroke')


# In[8]:


data=data[pd.isnull(data['postprocedural_stroke'])==False]


# In[9]:


data=data.reset_index(drop=True)


# # 数据处理

# In[10]:


for col in data.columns:
    try:
        data[col]=data[col].astype('float32')
    except:
        pass


# In[12]:


data.admission_age=data.admission_age.map(lambda x: 90 if x=='> 89' else x).astype('float32')


# # 缺失值

# In[15]:


tmp=data.isnull().sum()/data.shape[0]
tmp=tmp.sort_values(ascending=False)
tmp


# In[16]:


from sklearnex import patch_sklearn
patch_sklearn() # 这个函数用于开启加速sklearn，出现如下语句就表示OK！


# In[17]:


#KNN均值替换
from sklearn.impute import KNNImputer
imputer = KNNImputer()
X=data.drop(columns=['postprocedural_stroke'],axis=1)
X=pd.DataFrame(imputer.fit_transform(X),columns=X.columns)


# # 异常值

# In[18]:


data=data.reset_index(drop=True)


# In[19]:


data=pd.concat([X,data['postprocedural_stroke']],axis=1)


# In[20]:


data=data[data['BMI']<100].reset_index(drop=True)


# In[21]:


# LOF异常值处理
from sklearn.neighbors import LocalOutlierFactor
detector = LocalOutlierFactor(n_neighbors=10) # 构造异常值识别器
data['LOF']=detector.fit_predict(data)


# In[22]:


data['LOF'].value_counts()


# In[23]:


#-1为异常值,去除异常值
data=data[data['LOF']==1]
data.drop(columns=['LOF'],axis=1,inplace=True)


# In[24]:


data=data.reset_index(drop=True)


# # 数据标准化

# In[25]:


cols=[i for i in data.columns if i not in y_list]


# In[26]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df=pd.DataFrame(ss.fit_transform(data[cols]),columns=cols)
df['postprocedural_stroke']=data['postprocedural_stroke']


# In[27]:


import joblib
joblib.dump(ss,'ss.pkl')


# # postprocedural_stroke

# In[28]:


name='postprocedural_stroke'


# ## 数据不平衡数据

# In[29]:


X=df[cols]
y=df['postprocedural_stroke']
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
X_resampled, y_resampled = SMOTEENN(random_state=1).fit_resample(X, y)


# In[30]:


len(cols)


# In[31]:


from collections import Counter
print(sorted(Counter(y_resampled).items()))


# In[33]:


lasso_svm_selcet=cols
lasso_svm_selcet


# ## 定义模型评估函数

# In[34]:


from sklearn.metrics import precision_score, recall_score, f1_score ,roc_curve, auc,confusion_matrix ,accuracy_score,roc_auc_score,auc,brier_score_loss
def try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2):
    print('Train:')

    precision = precision_score(y_train,y_pred_train1)
    recall = recall_score(y_train,y_pred_train1)
    f1score = f1_score(y_train, y_pred_train1)
    accuracy=accuracy_score(y_train, y_pred_train1)
    cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
    TP=cnf_matrix[1,1]  # 1-->1
    TN=cnf_matrix[0,0]  # 0-->0
    FP=cnf_matrix[0,1]  # 0-->1
    FN=cnf_matrix[1,0]  # 1-->0
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_train2)
    AUC = auc(fpr, tpr)
    print("AUC:      ", '%.4f'%float(AUC),"ACC: ", '%.4f'%float(accuracy),"F1：", '%.4f'%float(f1score),"Precision:", '%.4f'%float(precision),\
    "Recall:   ",'%.4f'%float(recall),"Sensitivity :   ",'%.4f'%float(TP/(TP+FN)),"Specificity:   ",'%.4f'%float(TN/(FP+TN)))
    print('Model Train Report: \n',metrics.classification_report(y_train,y_pred_train1,digits=4))
    print('*'*50)
    print('Test:')

    precision = precision_score(y_test,y_pred_test1)
    recall = recall_score(y_test,y_pred_test1)
    f1score = f1_score(y_test, y_pred_test1)
    accuracy=accuracy_score(y_test, y_pred_test1)
    cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
    TP=cnf_matrix[1,1]  # 1-->1
    TN=cnf_matrix[0,0]  # 0-->0
    FP=cnf_matrix[0,1]  # 0-->1
    FN=cnf_matrix[1,0]  # 1-->0
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test2)
    AUC = auc(fpr, tpr)
    print("AUC:      ", '%.4f'%float(AUC),"ACC: ", '%.4f'%float(accuracy),"F1：", '%.4f'%float(f1score),"Precision:", '%.4f'%float(precision),\
    "Recall:   ",'%.4f'%float(recall),"Sensitivity:   ",'%.4f'%float(TP/(TP+FN)),"Specificity:   ",'%.4f'%float(TN/(FP+TN)))
    print('Model Test Report: \n',metrics.classification_report(y_test,y_pred_test1,digits=4))
    print('*'*50)
    print('Valid:')

    precision = precision_score(y_val,y_pred_val1)
    recall = recall_score(y_val,y_pred_val1)
    f1score = f1_score(y_val, y_pred_val1)
    accuracy=accuracy_score(y_val, y_pred_val1)
    cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
    TP=cnf_matrix[1,1]  # 1-->1
    TN=cnf_matrix[0,0]  # 0-->0
    FP=cnf_matrix[0,1]  # 0-->1
    FN=cnf_matrix[1,0]  # 1-->0
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_val2)
    AUC = auc(fpr, tpr)
    print("AUC:      ", '%.4f'%float(AUC),"ACC: ", '%.4f'%float(accuracy),"F1：", '%.4f'%float(f1score),"Precision:", '%.4f'%float(precision),\
    "Recall:   ",'%.4f'%float(recall),"Sensitivity:   ",'%.4f'%float(TP/(TP+FN)),"Specificity:   ",'%.4f'%float(TN/(FP+TN)))
    print('Model Valid Report: \n',metrics.classification_report(y_val,y_pred_val1,digits=4))
import itertools
def plot_roc(k,y_pred_undersample_score,labels_test,classifiers,color,title):
    fpr, tpr, thresholds = metrics.roc_curve(labels_test.values.ravel(),y_pred_undersample_score)
    roc_auc = metrics.auc(fpr,tpr)
    plt.figure(figsize=(20,16))
    plt.figure(k)
    plt.title(title)
    plt.plot(fpr, tpr, 'b',color=color,label='%s AUC = %0.4f'% (classifiers,roc_auc))
    plt.legend(loc='lower right',fontsize=10)
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    # plt.figure(figsize=(12,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",color="white" if cm[i, j] > thresh else "black",fontsize = 10,weight = 'heavy')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def plot_confusion_matrix2(cm, classes,title='Confusion matrix',cmap='red',fontsize=15):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize = fontsize)
    plt.xticks([])
    plt.yticks([])
    # plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=0)
    # plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.0%}'.format(cm[i, j]), horizontalalignment="center",color="black",fontsize = 15,weight = 'heavy')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.model_selection import KFold
# 设置k-fold交叉验证

from sklearn import metrics
def model_cv_score1(model,X,y,random_state=3,a=0.5):
    kfold = KFold(n_splits=5,random_state=random_state,shuffle=True)
    # 对数据进行交叉验证
    train_acc_list=[]
    test_acc_list=[]
    train_Precision_list=[]
    test_Precision_list=[]
    train_recall_list=[]
    test_recall_list=[]
    train_f1_list=[]
    test_f1_list=[]
    train_auc_list=[]
    test_auc_list=[]
    for i,(train_index, test_index) in enumerate(kfold.split(X)):
        # 获取训练集和测试集
        X_train1, X_test1 = X.loc[train_index], X.loc[test_index]
        y_train1, y_test1 = y.loc[train_index], y.loc[test_index]
        model.fit(X_train1, y_train1)                        #打乱标签
        # 在测试集上进行预测
        # y_pred_test1 = model.predict(X_test1)
        # y_pred_train1 = model.predict(X_train1)
        y_pred_test2 = model.predict_proba(X_test1)[:,1]
        y_pred_train2 = model.predict_proba(X_train1)[:,1]
        y_pred_test1=[int(i>a) for i in  y_pred_test2]
        y_pred_train1=[int(i>a) for i in  y_pred_train2]
        # 输出模型的准确率
        train_acc= metrics.accuracy_score(y_train1,y_pred_train1)
        test_acc=metrics.accuracy_score(y_test1,y_pred_test1)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # 输出模型的精度率
        train_Precision= metrics.precision_score(y_train1,y_pred_train1)
        test_Precision=metrics.precision_score(y_test1,y_pred_test1)
        train_Precision_list.append(train_Precision)
        test_Precision_list.append(test_Precision)    
        # 输出模型的召回率
        train_recall= metrics.recall_score(y_train1,y_pred_train1)
        test_recall=metrics.recall_score(y_test1,y_pred_test1)
        train_recall_list.append(train_recall)
        test_recall_list.append(test_recall)  
        # 输出模型的F1
        train_f1= metrics.f1_score(y_train1,y_pred_train1)
        test_f1=metrics.f1_score(y_test1,y_pred_test1)
        train_f1_list.append(train_f1)
        test_f1_list.append(test_f1)   
        # 输出模型AUC
        train_auc= metrics.roc_auc_score(y_train1,y_pred_train2)
        test_auc=metrics.roc_auc_score(y_test1,y_pred_test2)
        train_auc_list.append(train_auc)
        test_auc_list.append(test_auc)   
        print('Fold %s'%(i+1),'*'*50)
        print("train ACC:", train_acc,"train Precision:", train_Precision,"train Recall:", train_recall,"train F1:", train_f1,"train AUC:", train_auc)
        print("test ACC:", test_acc,"test Precision:", test_Precision,"test Recall:", test_recall,"test F1:", test_f1,"test AUC:", test_auc)
        print('\n')
    print('cross validation','*'*50)
    print('train Mean ACC',np.array(train_acc_list).mean(),'train Mean Precision',np.array(train_Precision_list).mean(),'train Mean Recall',np.array(train_recall_list).mean(),'train Mean F1',np.array(train_f1_list).mean(),'train Mean AUC',np.array(train_auc_list).mean())
    print('test Mean ACC',np.array(test_acc_list).mean(),'test Mean Precision',np.array(test_Precision_list).mean(),'test Mean Recall',np.array(test_recall_list).mean(),'test Mean F1',np.array(test_f1_list).mean(),'test Mean AUC',np.array(test_auc_list).mean())
    print('\n')
    print('\n')
    return np.array(train_acc_list),np.array(train_Precision_list),np.array(train_recall_list),np.array(train_f1_list),np.array(train_auc_list),\
            np.array(test_acc_list),np.array(test_Precision_list),np.array(test_recall_list),np.array(test_f1_list),np.array(test_auc_list)


# In[35]:


# for i in ['stroke','atrial_fibrillation','chart_order_beta_blockers','race','creatinine_Pre','heart_rate_Post','wbc_Pre']:
#     if i in lasso_svm_selcet:
#         lasso_svm_selcet.remove(i)


# In[36]:


# 数据分割
# 7比3划分训练集，测试集，设置随机种子random_state，保证实验能够复现
from sklearn.model_selection import train_test_split
X_train, _x, y_train, _y = train_test_split(X_resampled[lasso_svm_selcet],y_resampled,test_size=0.3,random_state=1)
X_test, X_val, y_test, y_val = train_test_split(_x,_y,test_size=0.333,random_state=1)
print(X_train.shape,X_test.shape,X_val.shape)


# In[37]:


from sklearn.model_selection import cross_val_score,StratifiedKFold,LeaveOneOut,KFold
from sklearn.model_selection import learning_curve,validation_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,train_sizes=np.linspace(.05, 1., 20), verbose=0,plot=True):
    train_sizes, train_scores, test_scores = \
    learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, verbose=verbose,scoring='roc_auc')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if plot:
        plt.figure(figsize=(10,5),dpi=120)
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        # plt.xlabel("train_size")
        plt.ylabel("score")
        plt.gca().invert_yaxis()
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="green", label="train_score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="red", label="cv_score")
    
        plt.legend(loc="best")
        # plt.ylim(1,0.7)
        plt.draw()
        plt.gca().invert_yaxis()
        # plt.show()
    
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


# ## LR

# ### 默认参数

# In[39]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=1,C=1e-7,class_weight='balanced')
clf.fit(X_train,y_train)
threshold=0.5
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# ### GA调参

# In[40]:


class GAIndividual:
 
    '''
    individual of genetic algorithm
    '''
 
    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
 
    def generate(self):
        '''
        generate a random chromsome for genetic algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]
 
    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = clfResult(self.vardim, self.chrom, self.bound)
import random
import copy
 
np.set_printoptions(suppress=True)
class GeneticAlgorithm:
 
    '''
    The class for genetic algorithm
    '''
 
    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        param: algorithm required parameters, it is a list which is consisting of crossover rate, mutation rate, alpha
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 3))
        self.params = params
 
    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)
 
    def evaluate(self):
        '''
        evaluation of the population fitnesses
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness
 
    def solve(self):
        '''
        evolution process of genetic algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluate()
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.maxfitness = np.max(self.fitness)
        
        self.trace[self.t, 0] =  self.best.fitness
        self.trace[self.t, 1] =  self.avefitness
        self.trace[self.t, 2] =  self.maxfitness
        print("Generation %d: optimal function value is: %f; average function value is %f;max function value is %f"% (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1],self.trace[self.t, 2]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.selectionOperation()
            self.crossoverOperation()
            self.mutationOperation()
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.maxfitness = np.max(self.fitness)
            
            self.trace[self.t, 0] =  self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            self.trace[self.t, 2] =  self.maxfitness
            print("Generation %d: optimal function value is: %f; average function value is %f;max function value is %f"% (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1],self.trace[self.t, 2]))
 
        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print ("Optimal solution is:")
        print (self.best.chrom)
 
    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        '''
        newpop = []
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.sizepop, 1))
 
        sum1 = 0.
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness
            sum1 = accuFitness[i]
 
        for i in range(0, self.sizepop):
            r = random.random()
            idx = 0
            for j in range(0, self.sizepop - 1):
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        self.population = newpop
 
    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()
            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):
                    newpop[i].chrom[j] = newpop[i].chrom[
                        j] * self.params[2] + (1 - self.params[2]) * newpop[i + 1].chrom[j]
                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                        (1 - self.params[2]) * newpop[i].chrom[j]
        self.population = newpop
 
    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] - (newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * (1 - random.random() ** (1 - self.t / self.MAXGEN))
                else:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] + (self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * (1 - random.random() ** (1 - self.t / self.MAXGEN))
        self.population = newpop


# In[43]:


##10.adding GA
def clfResult(vardim, x, bound):
    c=float(x[0])
    print("C:",round(c,6))     
    clf = LogisticRegression(C=c,random_state=1,class_weight='balanced') 
    clf.fit(X_train, y_train)
    predictval=clf.predict_proba(X_test)[:,1]
    print("ACC = ",metrics.roc_auc_score(y_test,predictval)) # R2
    return metrics.roc_auc_score(y_test,predictval)
bound = (np.array([[1e-8],[1e-6]]))
ga = GeneticAlgorithm(19, 1, bound, 1, [0.75, 0.25, 0.5])
ga.solve()


# ### 最优参数模型

# In[44]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=1,C=0.000001,class_weight='balanced')
clf.fit(X_train,y_train)
threshold=0.5
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[45]:


joblib.dump(clf,'./%s/clf_lr.pkl'%name)


# In[46]:


# 函数用来计算最佳阙值
def calculate_best_threshold(y, y_scores):
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][0],j_ordered[-1][1] # 返回最佳阙值
def bootstrap_auc(y, pred, classes, bootstraps = 1000, fold_size = 1000):
    statistics_auc = np.zeros((len(classes), bootstraps))
    statistics_acc = np.zeros((len(classes), bootstraps))
    statistics_f1 = np.zeros((len(classes), bootstraps))
    statistics_precision = np.zeros((len(classes), bootstraps))
    statistics_recall = np.zeros((len(classes), bootstraps))
    statistics_sens = np.zeros((len(classes), bootstraps))
    statistics_spec = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            j_ordered,threshold = calculate_best_threshold(y_sample, pred_sample)
            
            statistics_auc[c][i] = metrics.roc_auc_score(y_sample, pred_sample)
            y_pred=[int(i>threshold) for i in pred_sample]
            statistics_acc[c][i] = metrics.accuracy_score(y_sample,y_pred)
            statistics_f1[c][i] = metrics.f1_score(y_sample,y_pred)
            statistics_precision[c][i] = metrics.precision_score(y_sample,y_pred)
            statistics_recall[c][i] = metrics.recall_score(y_sample,y_pred)   
            cnf_matrix=metrics.confusion_matrix(y_sample,y_pred)
            TP=cnf_matrix[1,1]  # 1-->1
            TN=cnf_matrix[0,0]  # 0-->0
            FP=cnf_matrix[0,1]  # 0-->1
            FN=cnf_matrix[1,0]  # 1-->0            
            statistics_sens[c][i]=TP/(TP+FN)
            statistics_spec[c][i]=TN/(FP+TN)       
    return statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LR Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# In[47]:


statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LR Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# In[48]:


statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LR Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ### 模型评估

# In[49]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='LR Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='LR Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='LR Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/LR-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()
plt.figure(figsize=(8,5), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='LR Train',cmap='tab20')
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='LR Test',cmap='Greens_r')
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='LR Valid',cmap='Oranges_r')
plt.tight_layout()
plt.savefig('./%s/LR-confusion_matrix2.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[50]:


clf=joblib.load('./%s/clf_lr.pkl'%name)
train_acc_list_lr,train_Precision_list_lr,train_recall_list_lr,train_f1_list_lr,train_auc_list_lr,\
test_acc_list_lr,test_Precision_list_lr,test_recall_list_lr,test_f1_list_lr,test_auc_list_lr=model_cv_score1(clf,X_resampled[lasso_svm_selcet],y_resampled,a=0.5,random_state=3)


# In[51]:


clf=joblib.load('./%s/clf_lr.pkl'%name)
cv = KFold(n_splits=3,shuffle=True,random_state=0)
plot_learning_curve(clf, u" learning_curve", np.array(X_train), np.array(y_train),cv=cv)
plt.savefig('./%s/LR-CV-学习曲线.jpg'%name,dpi=600,bbox_inches = 'tight')
plt.show()


# ## lightgbm

# ### 默认参数

# In[52]:


import lightgbm as lgb
threshold=0.5
clf=lgb.LGBMClassifier(random_state=1,max_depth=1,n_estimators=1)
clf.fit(X_train,y_train)
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# ### GA调参

# In[53]:


##10.adding GA
def clfResult(vardim, x, bound):
    max_depth=round(x[0])
    n_estimators=round(x[1])
    print("max_depth:",round(max_depth),'n_estimators:',round(n_estimators))     
    clf = lgb.LGBMClassifier(random_state=1,max_depth=max_depth,n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    predictval=clf.predict_proba(X_test)[:,1]
    print("ACC = ",metrics.roc_auc_score(y_test,predictval)) # R2
    return metrics.roc_auc_score(y_test,predictval)
bound = (np.array([[1,1],[1,30]]))
ga = GeneticAlgorithm(19, 2, bound, 2, [0.75, 0.25, 0.5])
ga.solve()


# ### 最优参数模型

# In[54]:


import lightgbm as lgb
threshold=0.48
clf=lgb.LGBMClassifier(random_state=1,max_depth=1,n_estimators=29)
clf.fit(X_train,y_train)
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[55]:


joblib.dump(clf,'./%s/clf_lgb.pkl'%name)


# ### 模型评估

# In[56]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='lgb Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='lgb Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='lgb Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/lgb-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()
plt.figure(figsize=(8,5), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='lgb Train',cmap='tab20')
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='lgb Test',cmap='Greens_r')
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='lgb Valid',cmap='Oranges_r')
plt.tight_layout()
plt.savefig('./%s/lgb-confusion_matrix2.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[57]:


clf=joblib.load('./%s/clf_lgb.pkl'%name)
train_acc_list_lgb,train_Precision_list_lgb,train_recall_list_lgb,train_f1_list_lgb,train_auc_list_lgb,\
test_acc_list_lgb,test_Precision_list_lgb,test_recall_list_lgb,test_f1_list_lgb,test_auc_list_lgb=model_cv_score1(clf,X_resampled[lasso_svm_selcet],y_resampled,a=0.5,random_state=3)


# In[58]:


clf=joblib.load('./%s/clf_lgb.pkl'%name)
cv = KFold(n_splits=3,shuffle=True,random_state=3)
plot_learning_curve(clf, u" learning_curve", np.array(X_train), np.array(y_train),cv=cv)
plt.savefig('./%s/lgb-CV-学习曲线.jpg'%name,dpi=600,bbox_inches = 'tight')
plt.show()


# In[59]:


clf=joblib.load('./%s/clf_lgb.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1],bootstraps=1000)
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LGB Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1],bootstraps=1000)
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LGB Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1],bootstraps=1000)
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LGB Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## catboost

# ### 默认参数

# In[60]:


import catboost as cgb
clf=cgb.CatBoostClassifier(random_seed=1,depth=1,iterations=5,verbose=0)
clf.fit(X_train,y_train)
threshold=0.5
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# ### GA调参

# In[64]:


##10.adding GA
def clfResult(vardim, x, bound):
    max_depth=round(x[0])
    n_estimators=round(x[1])
    print("max_depth:",round(max_depth),'n_estimators:',round(n_estimators))     
    clf =cgb.CatBoostClassifier(random_seed=1,depth=max_depth,iterations=n_estimators,verbose=0)
    clf.fit(X_train, y_train)
    predictval=clf.predict_proba(X_test)[:,1]
    print("ACC = ",metrics.roc_auc_score(y_test,predictval)) # R2
    return metrics.roc_auc_score(y_test,predictval)
bound = (np.array([[1,1],[1,7]]))
ga = GeneticAlgorithm(19, 2, bound, 2, [0.75, 0.25, 0.5])
ga.solve()


# ### 最优参数模型

# In[65]:


import lightgbm as lgb
clf=cgb.CatBoostClassifier(random_seed=1,depth=1,iterations=7,verbose=0)
clf.fit(X_train,y_train)
threshold=0.5
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[66]:


joblib.dump(clf,'./%s/clf_cgb.pkl'%name)


# ### 模型评估

# In[67]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='cgb Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='cgb Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='cgb Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/cgb-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()
plt.figure(figsize=(8,5), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='cgb Train',cmap='tab20')
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='cgb Test',cmap='Greens_r')
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='cgb Valid',cmap='Oranges_r')
plt.tight_layout()
plt.savefig('./%s/cgb-confusion_matrix2.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[68]:


clf=joblib.load('./%s/clf_cgb.pkl'%name)
train_acc_list_cgb,train_Precision_list_cgb,train_recall_list_cgb,train_f1_list_cgb,train_auc_list_cgb,\
test_acc_list_cgb,test_Precision_list_cgb,test_recall_list_cgb,test_f1_list_cgb,test_auc_list_cgb=model_cv_score1(clf,X_resampled[lasso_svm_selcet],y_resampled,a=0.45)


# In[69]:


clf=joblib.load('./%s/clf_cgb.pkl'%name)
cv = KFold(n_splits=3,shuffle=True,random_state=291)
plot_learning_curve(clf, u" learning_curve", np.array(X_train), np.array(y_train),cv=cv)
plt.savefig('./%s/catboost-CV-学习曲线.jpg'%name,dpi=600,bbox_inches = 'tight')
plt.show()


# In[70]:


clf=joblib.load('./%s/clf_cgb.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("Catboost Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("Catboost Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("Catboost Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## GA-catboost-lr

# ### GA调参

# In[71]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
def clfResult(vardim, x, bound):
    max_depth=round(x[0])
    n_estimators=round(x[1])
    C = x[2]
    print("max_depth:",round(max_depth),'n_estimators:',round(n_estimators),'C:',C)     
    clf1 = LogisticRegression(random_state=1,C=C)
    clf2 =cgb.CatBoostClassifier(random_seed=1,depth=max_depth,iterations=n_estimators,verbose=0)
    estimators = [
    ('LR',     clf1),
    ('Catboost',    clf2),

    ]
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(random_state=1),n_jobs=-1,cv=3
    )
    clf.fit(X_train, y_train)
    predictval=clf.predict_proba(X_test)[:,1]
    print("ACC = ",metrics.roc_auc_score(y_test,predictval)) # R2
    return metrics.roc_auc_score(y_test,predictval)
bound = (np.array([[1,1,0.0001],[1,3,0.001]]))
ga = GeneticAlgorithm(19, 3, bound, 2, [0.75, 0.25, 0.5])
ga.solve()


# ### 最优参数模型

# In[72]:


from sklearn.ensemble import StackingClassifier
clf1 = LogisticRegression(random_state=1,C=0.00096458)
clf2 =cgb.CatBoostClassifier(random_seed=1,depth=1,iterations=3,verbose=0)
estimators = [
('LR',     clf1),
('Catboost',    clf2),

]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(random_state=1),n_jobs=-1,cv=3
)
clf.fit(X_train, y_train)


# In[73]:


y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
print('Stacking：\n')
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[74]:


joblib.dump(clf,'./%s/clf_GA-catboost-LR.pkl'%name)


# ### 模型评估

# In[75]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-catboost-LR Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-catboost-LR Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-catboost-LR Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/GA-catboost-LR-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()
plt.figure(figsize=(8,5), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-catboost-LR Train',cmap='tab20')
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-catboost-LR Test',cmap='Greens_r')
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-catboost-LR Valid',cmap='Oranges_r')
plt.tight_layout()
plt.savefig('./%s/GA-catboost-LR-confusion_matrix2.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[76]:


clf=joblib.load('./%s/clf_GA-catboost-LR.pkl'%name)
train_acc_list_stacking1,train_Precision_list_stacking1,train_recall_list_stacking1,train_f1_list_stacking1,train_auc_list_stacking1,\
test_acc_list_stacking1,test_Precision_list_stacking1,test_recall_list_stacking1,test_f1_list_stacking1,test_auc_list_stacking1=\
model_cv_score1(clf,X_resampled[lasso_svm_selcet],y_resampled,random_state=22)


# In[77]:


clf=joblib.load('./%s/clf_GA-catboost-LR.pkl'%name)
cv = KFold(n_splits=3,shuffle=True,random_state=291)
plot_learning_curve(clf, u" learning_curve", np.array(X_train), np.array(y_train),cv=cv)
plt.savefig('./%s/GA-catboost-LR-CV-学习曲线.jpg'%name,dpi=600,bbox_inches = 'tight')
plt.show()


# In[78]:


clf=joblib.load('./%s/clf_GA-catboost-LR.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-catboost-LR Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-catboost-LR Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-catboost-LR Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## GA-lightgbm-lr

# ### GA调参

# In[79]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
def clfResult(vardim, x, bound):
    max_depth=round(x[0])
    n_estimators=round(x[1])
    C = x[2]
    print("max_depth:",round(max_depth),'n_estimators:',round(n_estimators),'C:',C)     
    clf1 = LogisticRegression(random_state=1,C=C)
    clf2 =lgb.LGBMClassifier(random_seed=1,max_depth=max_depth,n_estimators=n_estimators,verbose=0)
    estimators = [
    ('LR',     clf1),
    ('lgb',    clf2),

    ]
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(random_state=1),n_jobs=-1,cv=3
    )
    clf.fit(X_train, y_train)
    predictval=clf.predict_proba(X_test)[:,1]
    print("ACC = ",metrics.roc_auc_score(y_test,predictval)) # R2
    return metrics.roc_auc_score(y_test,predictval)
bound = (np.array([[1,1,0.0001],[1,10,0.01]]))
ga = GeneticAlgorithm(19, 3, bound, 2, [0.75, 0.25, 0.5])
ga.solve()


# ### 最优参数模型

# In[80]:


clf1 = LogisticRegression(random_state=1,C=0.00940114)
clf2 =lgb.LGBMClassifier(random_seed=1,max_depth=1,n_estimators=3,verbose=0)
estimators = [
('LR',     clf1),
('lgb',    clf2),

]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(random_state=1),n_jobs=-1,cv=3
)
clf.fit(X_train, y_train)


# In[81]:


threshold=0.5
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[82]:


joblib.dump(clf,'./%s/clf_GA-lightgbm-LR.pkl'%name)


# ### 模型评估

# In[83]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-lightgbm-LR Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-lightgbm-LR Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-lightgbm-LR Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/GA-lightgbm-LR-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()
plt.figure(figsize=(8,5), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-lightgbm-LR Train',cmap='tab20')
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-lightgbm-LR Test',cmap='Greens_r')
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-lightgbm-LR Valid',cmap='Oranges_r')
plt.tight_layout()
plt.savefig('./%s/GA-lightgbm-LR-confusion_matrix2.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[84]:


clf=joblib.load('./%s/clf_GA-lightgbm-LR.pkl'%name)
train_acc_list_stacking2,train_Precision_list_stacking2,train_recall_list_stacking2,train_f1_list_stacking2,train_auc_list_stacking2,\
test_acc_list_stacking2,test_Precision_list_stacking2,test_recall_list_stacking2,test_f1_list_stacking2,test_auc_list_stacking2=\
model_cv_score1(clf,X_resampled[lasso_svm_selcet],y_resampled,random_state=22,a=0.6)


# In[85]:


clf=joblib.load('./%s/clf_GA-lightgbm-LR.pkl'%name)
cv = KFold(n_splits=3,shuffle=True,random_state=291)
plot_learning_curve(clf, u" learning_curve", np.array(X_train), np.array(y_train),cv=cv)
plt.savefig('./%s/GA-lightgbm-LR-CV-学习曲线.jpg'%name,dpi=600,bbox_inches = 'tight')
plt.show()


# In[86]:


clf=joblib.load('./%s/clf_GA-lightgbm-LR.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-lightgbm-LR Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-lightgbm-LR Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-lightgbm-LR Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## GA-lightgbm-catboost-LR

# ### GA调参

# In[87]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
def clfResult(vardim, x, bound):
    max_depth=round(x[0])
    n_estimators=round(x[1])
    C = x[2]
    depth=round(x[3])
    iterations=round(x[4])    
    print("max_depth:",round(max_depth),'n_estimators:',round(n_estimators),'C:',C,"depth:",round(depth),'iterations:',round(iterations))     
    clf1 = LogisticRegression(random_state=1,C=C)
    clf2 =lgb.LGBMClassifier(random_seed=1,max_depth=max_depth,n_estimators=n_estimators,verbose=0)
    clf3 =cgb.CatBoostClassifier(random_seed=1,depth=depth,iterations=iterations,verbose=0)
    estimators = [
    ('LR',     clf1),
    ('lgb',    clf2),
    ('catboost',    clf3),
    ]
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(random_state=1),n_jobs=-1,cv=3
    )
    clf.fit(X_train, y_train)
    predictval=clf.predict_proba(X_test)[:,1]
    print("ACC = ",metrics.roc_auc_score(y_test,predictval)) # R2
    return metrics.roc_auc_score(y_test,predictval)
bound = (np.array([[1,1,0.0001,1,1],[2,30,1,2,30]]))
ga = GeneticAlgorithm(19, 5, bound, 2, [0.75, 0.25, 0.5])
ga.solve()


# ### 最优参数模型

# In[88]:


clf1 = LogisticRegression(random_state=1,C=0.93302542  )
clf2 =lgb.LGBMClassifier(random_seed=1,max_depth=1,n_estimators=26,verbose=0)
clf3 =cgb.CatBoostClassifier(random_seed=1,depth=2,iterations=25,verbose=0)
estimators = [
('LR',     clf1),
('lgb',    clf2),
('catboost',    clf3),
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(random_state=1),n_jobs=-1,cv=3
)
clf.fit(X_train, y_train)


# In[89]:


y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[90]:


joblib.dump(clf,'./%s/clf_GA-lightgbm-catboost-LR.pkl'%name)


# ### 模型评估

# In[91]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-lightgbm-catboost-LR Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-lightgbm-catboost-LR Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='GA-lightgbm-catboost-LR Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/GA-lightgbm-catboost-LR-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()
plt.figure(figsize=(8,5), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-lightgbm-catboost-LR Train',cmap='tab20',fontsize=12)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-lightgbm-catboost-LR Test',cmap='Greens_r',fontsize=12)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
cnf_matrix=cnf_matrix/cnf_matrix.sum(axis=0)
plot_confusion_matrix2(cnf_matrix,[0,1],title='GA-lightgbm-catboost-LR Valid',cmap='Oranges_r',fontsize=12)
plt.tight_layout()
plt.savefig('./%s/GA-lightgbm-catboost-LR-confusion_matrix2.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[92]:


clf=joblib.load('./%s/clf_GA-lightgbm-catboost-LR.pkl'%name)
train_acc_list_stacking3,train_Precision_list_stacking3,train_recall_list_stacking3,train_f1_list_stacking3,train_auc_list_stacking3,\
test_acc_list_stacking3,test_Precision_list_stacking3,test_recall_list_stacking3,test_f1_list_stacking3,test_auc_list_stacking3=\
model_cv_score1(clf,X_resampled[lasso_svm_selcet],y_resampled,random_state=1,a=0.45)


# In[93]:


clf=joblib.load('./%s/clf_GA-lightgbm-catboost-LR.pkl'%name)
cv = KFold(n_splits=3,shuffle=True,random_state=291)
plot_learning_curve(clf, u" learning_curve", np.array(X_train), np.array(y_train),cv=cv)
plt.savefig('./%s/GA-lightgbm-catboost-LR-CV-学习曲线.jpg'%name,dpi=600,bbox_inches = 'tight')
plt.show()


# In[94]:


clf=joblib.load('./%s/clf_GA-lightgbm-catboost-LR.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-lightgbm-catboost-LR Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-lightgbm-catboost-LR Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')
statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list1=['AUC','ACC','F1','Precision','Recall','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("GA-lightgbm-catboost-LR Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## ROC曲线

# In[95]:


color=['red','blue','green','tomato','orange','darkred']
name_list=['LR','Lightgbm','Catboost','GA-catboost-LR','GA-lightgbm-LR','GA-lightgbm-catboost-LR']
name='postprocedural_stroke'
clf1=joblib.load('./%s/clf_lr.pkl'%name)
clf2=joblib.load('./%s/clf_lgb.pkl'%name)
clf3=joblib.load('./%s/clf_cgb.pkl'%name)
clf4=joblib.load('./%s/clf_GA-catboost-LR.pkl'%name)
clf5=joblib.load('./%s/clf_GA-lightgbm-LR.pkl'%name)
clf6=joblib.load('./%s/clf_GA-lightgbm-catboost-LR.pkl'%name)
model_list=[clf1,clf2,clf3,clf4,clf5,clf6]
# fig = plt.gcf()
# fig.set_size_inches(15,4)
plt.figure(figsize=(15,5), dpi=120)
plt.subplot(1,3,1)
for i,(name,model) in enumerate(zip(name_list,model_list)):
    plot_roc(1,model.predict_proba(X_train)[:,1],y_train,name,color[i],'Train  ROC curve')
plt.subplot(1,3,2)
for i,(name,model) in enumerate(zip(name_list,model_list)):
    plot_roc(1,model.predict_proba(X_test)[:,1],y_test,name,color[i],'Test  ROC curve')
plt.subplot(1,3,3)
for i,(name,model) in enumerate(zip(name_list,model_list)):
    plot_roc(1,model.predict_proba(X_val)[:,1],y_val,name,color[i],'Valid  ROC curve')
plt.tight_layout()
name='postprocedural_stroke'
plt.savefig('./%s/ROC-Curve.jpg'%name,dpi=600,bbox_inches = 'tight')
plt.show()


# ## 雷达图

# In[96]:


columns=['ACC','Precision','Recall','F1','AUC']
tmp1=pd.DataFrame([[test_acc_list_lr.mean(),test_Precision_list_lr.mean(),test_recall_list_lr.mean(),test_f1_list_lr.mean(),test_auc_list_lr.mean()],
[test_acc_list_lgb.mean(),test_Precision_list_lgb.mean(),test_recall_list_lgb.mean(),test_f1_list_lgb.mean(),test_auc_list_lgb.mean()],
[test_acc_list_cgb.mean(),test_Precision_list_cgb.mean(),test_recall_list_cgb.mean(),test_f1_list_cgb.mean(),test_auc_list_cgb.mean()],
[test_acc_list_stacking1.mean(),test_Precision_list_stacking1.mean(),test_recall_list_stacking1.mean(),test_f1_list_stacking1.mean(),test_auc_list_stacking1.mean()],
[test_acc_list_stacking2.mean(),test_Precision_list_stacking2.mean(),test_recall_list_stacking2.mean(),test_f1_list_stacking2.mean(),test_auc_list_stacking2.mean()],
[test_acc_list_stacking3.mean(),test_Precision_list_stacking3.mean(),test_recall_list_stacking3.mean(),test_f1_list_stacking3.mean(),test_auc_list_stacking3.mean()]
],columns=columns,index=name_list)
tmp1=tmp1.T
tmp1


# In[97]:


tmp1.to_excel('./%s/Model_comparison.xlsx'%name)


# In[99]:


plt.rcParams['axes.unicode_minus'] = False
# 构造数据
values1 = tmp1['LR'].values
values2 = tmp1['Lightgbm'].values
values3 = tmp1['Catboost'].values
values4 = tmp1['GA-catboost-LR'].values
values5 = tmp1['GA-lightgbm-LR'].values
values6 = tmp1['GA-lightgbm-catboost-LR'].values
N = len(values1)
# 设置雷达图的角度，用于平分切开一个圆面
angles=np.linspace(0, 2*np.pi, N, endpoint=False)
# 为了使雷达图一圈封闭起来，需要下面的步骤
values1=np.concatenate((values1,[values1[0]]))
values2=np.concatenate((values2,[values2[0]]))
values3=np.concatenate((values3,[values3[0]]))
values4=np.concatenate((values4,[values4[0]]))
values5=np.concatenate((values5,[values5[0]]))
values6=np.concatenate((values6,[values6[0]]))
angles=np.concatenate((angles,[angles[0]]))
# 绘图
fig=plt.figure(figsize=(12,8),dpi=120)
ax = fig.add_subplot(111, polar=True)
# 绘制折线图
ax.plot(angles, values1, 'o-', linewidth=2, label = 'LR')
ax.fill(angles, values1, alpha=0.25)
# 绘制第二条折线图
ax.plot(angles, values2, 'o-', linewidth=2, label = 'Lightgbm')
ax.fill(angles, values2, alpha=0.25)
# 绘制第三条折线图
ax.plot(angles, values3, 'o-', linewidth=2, label = 'Catboost')
ax.fill(angles, values3, alpha=0.25)
# 绘制第四条折线图
ax.plot(angles, values4, 'o-', linewidth=2, label = 'GA-catboost-LR')
ax.fill(angles, values4, alpha=0.25)
# 绘制第四条折线图
ax.plot(angles, values5, 'o-', linewidth=2, label = 'GA-lightgbm-LR')
ax.fill(angles, values5, alpha=0.25)
# 绘制第四条折线图
ax.plot(angles, values6, 'o-', linewidth=2, label = 'GA-lightgbm-catboost-LR')
ax.fill(angles, values6, alpha=0.25)
# 添加每个特征的标签
ax.set_thetagrids((angles * 180/np.pi)[:-1], tmp1.index.tolist())
# 设置雷达图的范围
ax.set_ylim(0.8,1)
# 添加标题
plt.title('Comparison of model cross-validation effect')
# 添加网格线
ax.grid(True)
plt.yticks(fontsize=12, color='k')
plt.xticks(fontsize=12, color='k')
# 设置图例
plt.legend(loc = 1,fontsize=8)
plt.savefig('./%s/Model_comparison_Radar_map.jpg'%name,dpi=600,bbox_inches = 'tight')
# 显示图形
plt.show()


# ## 模型预测

# In[101]:


test_data={'admission_age': 66.08898,
 'BMI': 32.098766,
 'atrial_fibrillation': 0.0,
 'valvular_disease': 0.0,
 'stroke': 0.0,
 'chronic_renal_failure': 0.0,
 'myocardial_infarct': 0.0,
 'congestive_heart_failure': 0.0,
 'peripheral_vascular_disease': 0.0,
 'hypertension': 1.0,
 'mbp_Pre': 62.0,
 'resp_rate_Pre': 19.0,
 'glucose_Pre': 126.0,
 'potassium_Pre': 4.6,
 'hemoglobin_Pre': 9.2,
 'mbp_Post': 62.0,
 'resp_rate_Post': 18.0,
 'glucose_Post': 126.0,
 'potassium_Post': 4.6,
 'hemoglobin_Post': 9.2}
test_data=pd.DataFrame(ss.transform(pd.DataFrame([test_data])[ss.feature_names_in_]),columns=ss.feature_names_in_)
test_data[lasso_svm_selcet]


# In[102]:


name='postprocedural_stroke'
clf1=joblib.load('./%s/clf_lr.pkl'%name)
clf2=joblib.load('./%s/clf_lgb.pkl'%name)
clf3=joblib.load('./%s/clf_cgb.pkl'%name)
clf4=joblib.load('./%s/clf_GA-catboost-LR.pkl'%name)
clf5=joblib.load('./%s/clf_GA-lightgbm-LR.pkl'%name)
clf6=joblib.load('./%s/clf_GA-lightgbm-catboost-LR.pkl'%name)
for i,(name,model) in enumerate(zip(name_list,model_list)):
    print(name,'Prediction class:',model.predict(test_data[lasso_svm_selcet])[0],'Prediction probability:',round(model.predict_proba(test_data[lasso_svm_selcet])[:,1][0]*10))

