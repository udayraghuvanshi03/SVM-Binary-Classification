import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm
import os

class SVM():
    def __init__(self,file):
        self.file=file
    # Extracting data

    def data_sep(self):
        if self.file=='dataset1.mat':
            dic_1 = scipy.io.loadmat(self.file)
            X_trn = dic_1['X_trn']
            Y_trn = dic_1['Y_trn']
            X_tst = dic_1['X_tst']
            Y_tst = dic_1['Y_tst']
            Y_trn = np.ravel(Y_trn)
            Y_tst = np.ravel(Y_tst)
            return X_trn, Y_trn, X_tst, Y_tst

        elif self.file=='dataset2.mat':
            dic_2=scipy.io.loadmat(self.file)
            X_trn=dic_2['xtrain']
            Y_trn=dic_2['ytrain']
            X_tst=dic_2['xtest']
            Y_tst=dic_2['ytest']
            Y_trn=np.ravel(Y_trn)
            Y_tst=np.ravel(Y_tst)
            return X_trn,Y_trn,X_tst,Y_tst

        else:
            print('Invalid input')

    def classification(self,X_trn,Y_trn,X_tst,Y_tst,kernel_func : str):
        if self.file == 'dataset1.mat':
            Y_tst = Y_tst.tolist()
            Y_trn = Y_trn.tolist()
            for i in range(len(Y_tst)):
                if Y_tst[i] == 0:
                    Y_tst[i] = -1
            for i in range(len(Y_trn)):
                if Y_trn[i] == 0:
                    Y_trn[i] = -1

        model = svm.SVC(kernel=str.lower(kernel_func))
        clf = model.fit(X_trn, Y_trn)
        if kernel_func=='linear':
            intercept=clf.intercept_
            coeff=clf.coef_
            c = -intercept / coeff[0][1]
            m = -coeff[0][0] / coeff[0][1]
            x_tr_min,x_tr_max = np.min(X_trn),np.max(X_trn)
            x1 = np.array([x_tr_min, x_tr_max])
            x2 = m * x1 + c
            plt.plot(x1, x2, 'k', ls='--',label='decision boundary')

        else:
            x_min, x_max = np.min(X_trn[:, 0]) - 1, np.max(X_trn[:, 0]) + 1
            y_min, y_max = np.min(X_trn[:, 1]) - 1, np.max(X_trn[:, 1]) + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            cs=plt.contour(xx,yy,Z,colors='black')
        Y_train_pred = model.predict(X_trn)
        Y_test_pred = model.predict(X_tst)
        # print(Y_train_pred)
        count_test = 0
        count_train = 0
        for i in range(len(Y_trn)):
            if Y_train_pred[i] == Y_trn[i]:
                count_train += 1
        train_acc = count_train / len(Y_trn)

        for i in range(len(Y_tst)):
            if Y_test_pred[i] == Y_tst[i]:
                count_test += 1
        test_acc = count_test / len(Y_tst)
        print('Training accuracy for', self.file, kernel_func, ': ', 100 * train_acc, '%')
        print('Test accuracy for', self.file, kernel_func, ': ', 100 * test_acc, '%')

        plt.scatter(X_trn[:, 0], X_trn[:, 1], color='green', marker='o', label='training data')
        tst_class_neg_1 = []
        tst_class_1 = []
        for i in range(len(Y_test_pred)):
            if Y_test_pred[i] == -1:
                tst_class_neg_1.append((X_tst[i][0], X_tst[i][1]))
            else:
                tst_class_1.append((X_tst[i][0], X_tst[i][1]))

        tst_neg_x_1, tst_neg_x_2 = zip(*tst_class_neg_1)
        tst_x_1, tst_x_2 = zip(*tst_class_1)
        plt.scatter(tst_neg_x_1, tst_neg_x_2, color='red', marker='x', label='class -1 (predicted)')
        plt.scatter(tst_x_1, tst_x_2, color='blue', marker='x', label='class 1 (predicted)')

        plt.xlabel('x_1')
        plt.ylabel('x_2')
        title = 'classification of ' + self.file + ' using '+ kernel_func + ' kernel'
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.show()

if __name__=='__main__':
    data_path=r'C:\Users\udayr\PycharmProjects\MLfiles\cs6140-hw03-dataset\cs6140-hw03-dataset'
    dataset_1=os.listdir(data_path)[0]
    dataset_2=os.listdir(data_path)[1]
    svm_1 = SVM(dataset_1)
    data_1= svm_1.data_sep()
    linear_1= svm_1.classification(data_1[0], data_1[1], data_1[2], data_1[3], 'linear')
    RBF_1=svm_1.classification(data_1[0], data_1[1], data_1[2], data_1[3], 'RBF')
    poly_1 = svm_1.classification(data_1[0], data_1[1], data_1[2], data_1[3], 'poly')

    svm_2=SVM(dataset_2)
    data_2=svm_2.data_sep()
    linear_2=svm_2.classification(data_2[0],data_2[1],data_2[2],data_2[3],'linear')
    RBF_2=svm_2.classification(data_2[0],data_2[1],data_2[2],data_2[3],'RBF')
    poly_2=svm_2.classification(data_2[0],data_2[1],data_2[2],data_2[3],'poly')
