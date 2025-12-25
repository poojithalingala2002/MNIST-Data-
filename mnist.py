import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV



from sklearn.utils import multiclass
class MNIST():
    def __init__(self,train_data,test_data):
        try:
            self.train_data=pd.read_csv(train_data)
            self.test_data=pd.read_csv(test_data)
            self.X_train=self.train_data.iloc[:,1:]
            self.y_train=self.train_data.iloc[:,0]
            self.X_test=self.test_data.iloc[:,1:]
            self.y_test=self.test_data.iloc[:,0]
        except Exception as e:
            ex_type,ex_msg,ex_line=sys.exc_info()
            print(f"Error Type : {ex_type} Error Message : {ex_msg} in Line : {ex_line.tb_lineno}")

    try:
        def svm_grid(self):
            print("----------SVM GRID SEARCH---------")
            self.X_sample=self.X_train.iloc[:600,:]
            self.y_sample=self.y_train.iloc[:600]
            self.parameters={'C':[0.1, 1, 10],
                        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                        'class_weight':[None, 'balanced']
                        }
            
            self.reg=SVC()
            self.grid_model=GridSearchCV(estimator=self.reg,
                                    param_grid=self.parameters,
                                    cv=5,
                                    scoring='accuracy'
                                    )
            self.grid_model.fit(self.X_sample,self.y_sample)
            print("Best Parameters we got :",self.grid_model.best_params_)
            self.best_params=self.grid_model.best_params_


            self.svm_grid_reg=SVC(C=self.best_params['C'],
                                kernel=self.best_params['kernel'],
                                class_weight=self.best_params['class_weight'],
                                probability=True
                                )
            self.svm_grid_reg.fit(self.X_train, self.y_train)
            pred=self.svm_grid_reg.predict(self.X_test)
            print(f'Test accuracy of tuned SVM: {accuracy_score(self.y_test, pred)}')
            print(f'confusion matrix of tuned SVM:\n{confusion_matrix(self.y_test, pred)}')
            print(f'classification report:\n{classification_report(self.y_test, pred)}')


            self.predictions=self.svm_grid_reg.predict_proba(self.X_test)
            self.svm_grid_fpr,self.svm_grid_tpr=self.roc()   
            self.plot1()     

        def plot1(self):
            plt.figure(figsize=(5,3))
            plt.plot([0,1],[0,1],'k--')
            plt.plot(self.knn_fpr,self.knn_tpr,label='knn')
            plt.plot(self.lg_fpr,self.lg_tpr,label='lg')
            plt.plot(self.nb_fpr,self.nb_tpr,label='nb')
            plt.plot(self.dt_fpr,self.dt_tpr,label='dt')
            plt.plot(self.rf_fpr,self.rf_tpr,label='rf')
            plt.plot(self.ada_fpr,self.ada_tpr,label='ada')
            plt.plot(self.gb_fpr,self.gb_tpr,label='gb')
            plt.plot(self.xgb_fpr,self.xgb_tpr,label='xgb')
            plt.plot(self.svm_fpr,self.svm_tpr,label='svm')
            plt.plot(self.svm_grid_fpr,self.svm_grid_tpr,label='svm_grid')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title("ROC Curve - ALL Models")
            plt.legend()
            plt.show()

        def algo(self):
            print('----------KNN---------')
            self.knn()
            print('----------lg---------')
            self.lg()
            print('----------nb---------')
            self.nb()
            print('----------dt---------')
            self.dt()
            print('----------rf---------')
            self.rf()
            print('----------ada---------')
            self.ada()
            print('----------gb---------')
            self.gb()
            print('----------xgb---------')
            self.xgb()
            print('----------svm---------')
            self.svm()
            self.plot()
            print('----------svm with grid search---------')
            self.svm_grid()

        def knn(self):
            self.knn_reg=KNeighborsClassifier(n_neighbors=5)
            self.knn_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of KNN :{accuracy_score(self.y_test,self.knn_reg.predict(self.X_test))}')
            print(f'confusion matrix of KNN:\n{confusion_matrix(self.y_test,self.knn_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.knn_reg.predict(self.X_test))}')
            self.predictions=self.knn_reg.predict_proba(self.X_test) 
            self.knn_fpr,self.knn_tpr=self.roc()

        def lg(self):
            self.lg_reg=LogisticRegression()
            self.lg_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of lg :{accuracy_score(self.y_test,self.lg_reg.predict(self.X_test))}')
            print(f'confusion matrix of lg:\n{confusion_matrix(self.y_test,self.lg_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.lg_reg.predict(self.X_test))}')
            self.predictions=self.lg_reg.predict_proba(self.X_test)
            self.lg_fpr,self.lg_tpr=self.roc()

        def nb(self):
            self.nb_reg=GaussianNB()
            self.nb_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of nb :{accuracy_score(self.y_test,self.nb_reg.predict(self.X_test))}')
            print(f'confusion matrix of nb:\n{confusion_matrix(self.y_test,self.nb_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.nb_reg.predict(self.X_test))}')
            self.predictions=self.nb_reg.predict_proba(self.X_test)
            self.nb_fpr,self.nb_tpr=self.roc()

        
        def dt(self):
            self.dt_reg=DecisionTreeClassifier(criterion='entropy')
            self.dt_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of dt :{accuracy_score(self.y_test,self.dt_reg.predict(self.X_test))}')
            print(f'confusion matrix of dt:\n{confusion_matrix(self.y_test,self.dt_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.dt_reg.predict(self.X_test))}')
            self.predictions=self.dt_reg.predict_proba(self.X_test)
            self.dt_fpr,self.dt_tpr=self.roc()

        
        def rf(self):
            self.rf_reg=RandomForestClassifier(n_estimators=5,criterion='entropy')
            self.rf_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of rf :{accuracy_score(self.y_test,self.rf_reg.predict(self.X_test))}')
            print(f'confusion matrix of rf:\n{confusion_matrix(self.y_test,self.rf_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.rf_reg.predict(self.X_test))}')
            self.predictions=self.rf_reg.predict_proba(self.X_test)
            self.rf_fpr,self.rf_tpr=self.roc()

        
        def ada(self):
            self.t=LogisticRegression()
            self.ada_reg=AdaBoostClassifier(estimator=self.t,n_estimators=5)
            self.ada_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of ada :{accuracy_score(self.y_test,self.ada_reg.predict(self.X_test))}')
            print(f'confusion matrix of ada:\n{confusion_matrix(self.y_test,self.ada_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.ada_reg.predict(self.X_test))}')
            self.predictions=self.ada_reg.predict_proba(self.X_test)
            self.ada_fpr,self.ada_tpr=self.roc()

        
        def gb(self):
            self.gb_reg=GradientBoostingClassifier(n_estimators=5)
            self.gb_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of gb :{accuracy_score(self.y_test,self.gb_reg.predict(self.X_test))}')
            print(f'confusion matrix of gb:\n{confusion_matrix(self.y_test,self.gb_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.gb_reg.predict(self.X_test))}')
            self.predictions=self.gb_reg.predict_proba(self.X_test)
            self.gb_fpr,self.gb_tpr=self.roc()

        
        def xgb(self):
            self.xgb_reg=XGBClassifier()
            self.xgb_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of xgb :{accuracy_score(self.y_test,self.xgb_reg.predict(self.X_test))}')
            print(f'confusion matrix of xgb:\n{confusion_matrix(self.y_test,self.xgb_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.xgb_reg.predict(self.X_test))}')
            self.predictions=self.xgb_reg.predict_proba(self.X_test)
            self.xgb_fpr,self.xgb_tpr=self.roc()

        
        def svm(self):
            self.svm_reg=SVC(kernel='rbf',probability=True)
            self.svm_reg.fit(self.X_train,self.y_train)
            print(f'Test accuracy of svm :{accuracy_score(self.y_test,self.svm_reg.predict(self.X_test))}')
            print(f'confusion matrix of svm:\n{confusion_matrix(self.y_test,self.svm_reg.predict(self.X_test))}')
            print(f'classification report:\n{classification_report(self.y_test,self.svm_reg.predict(self.X_test))}')
            self.predictions=self.svm_reg.predict_proba(self.X_test)
            self.svm_fpr,self.svm_tpr=self.roc()

        def roc(self):
            self.y_test_bin = label_binarize(self.y_test, classes=range(10))
            self.fpr_list=[]
            self.tpr_list=[]
            for i in range(10):
                self.fpr,self.tpr,self.thr=roc_curve(self.y_test_bin[:,i],self.predictions[:,i])
                self.fpr_list.append(self.fpr)
                self.tpr_list.append(self.tpr)
            self.all_fpr=np.unique(np.concatenate(self.fpr_list))
            self.mean_tpr=np.zeros_like(self.all_fpr) 
            for i in range(10):
                self.mean_tpr += np.interp(self.all_fpr, self.fpr_list[i], self.tpr_list[i])
            self.mean_tpr /= 10
            return self.all_fpr,self.mean_tpr 

        
        def plot(self):
            plt.figure(figsize=(5,3))
            plt.plot([0,1],[0,1],'k--')
            plt.plot(self.knn_fpr,self.knn_tpr,label='knn')
            plt.plot(self.lg_fpr,self.lg_tpr,label='lg')
            plt.plot(self.nb_fpr,self.nb_tpr,label='nb')
            plt.plot(self.dt_fpr,self.dt_tpr,label='dt')
            plt.plot(self.rf_fpr,self.rf_tpr,label='rf')
            plt.plot(self.ada_fpr,self.ada_tpr,label='ada')
            plt.plot(self.gb_fpr,self.gb_tpr,label='gb')
            plt.plot(self.xgb_fpr,self.xgb_tpr,label='xgb')
            plt.plot(self.svm_fpr,self.svm_tpr,label='svm')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title("ROC Curve - ALL Models")
            plt.legend()
            plt.show()
    except Exception as e:
        ex_type,ex_msg,ex_line=sys.exc_info()
        print(f"Error Type : {ex_type} Error Message : {ex_msg} in Line : {ex_line.tb_lineno}")

if __name__=="__main__":
  try:
    train_data=r"D:\Projects\mnist\mnist_train.csv"
    test_data=r"D:\Projects\mnist\mnist_test.csv"
    obj = MNIST(train_data,test_data)
    obj.algo()


  except Exception as e:
    ex_type,ex_msg,ex_line=sys.exc_info()

    print(f"Error Type : {ex_type} Error Message : {ex_msg} in Line : {ex_line.tb_lineno}")
