import keras
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.model_selection import cross_val_score

class ClassificationReport(keras.callbacks.Callback):
    #def __init__(self, logs={}):
        #print(train_data)

    def on_train_begin(self, logs={}):        
        self._data = []
        self._report = []
        self._names = [
            'Neurosis 1','Neurosis 2', 'Neurosis 3',
            'Responsible 1', 'Responsible 2', 'Responsible 3',
            'Agreeableness 1', 'Agreeableness 2', 'Agreeableness 3',
            'PassionForNewExperiences 1', 'PassionForNewExperiences 2', 'PassionForNewExperiences 3',
            'ExtroversionIntroversion 1', 'ExtroversionIntroversion 2', 'ExtroversionIntroversion 3'
        ]
        print("on_train_begin")

    #def on_batch_end(self, batch, logs={}):
        #print(batch)         
        #l = dir(batch)
        #print(l)        
        #print("on_batch_end")

    def on_epoch_end(self, batch, logs={}):
        print("on_epoch_end")

        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = self.model.predict(X_val)
        
        y_test = []
        y_pred = []
        
        #print(len(y_predict))
        #print(y_predict[0])
        print('\n step 01 \n')

        for i in range(len(y_predict)):
            target = []
            predict = []
            #print(str(i))
            for j in range(len(y_predict[i])):
                #print(str(j))
                predict.append(1)
                target.append(y_val[i][j])
                if y_predict[i][j] < 0.1 :
                   predict[j] = 0
            
            #print(target)
            #print(predict)
            y_test.append(target)
            y_pred.append(predict)
        
        print(y_test[0])
        print(y_pred[0])
        
        print('\n step 02 \n')

        confusion = multilabel_confusion_matrix(y_test, y_pred)        
        #print('Confusion Matrix\n')
        #print(confusion)
        

        print('\nClassification Report\n')               
        names = self._names
        report = classification_report(y_test, y_pred, target_names=names)
        
        print(report)
        f = open("ClassificationReport2.txt", "w")        
        f.write(report)
        f.write("\n ----------------------------------------------------------------------------- \n")
        f.close()

        n_classes = len(names)
        
        print('\n step 03 ROC \n')
        print(names)
        print(n_classes)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            #print('\n step 03.00 ROC \n')            
            y_i = np.array(y_test)[:, i]
            p_i = np.array(y_pred)[:, i]
            #print(y_i)
            #print(p_i)
            fpr[i], tpr[i], _ = roc_curve(y_i, p_i)            
            #print('\n step 03.01 ROC \n')
            fpr["micro"], tpr["micro"], _ = roc_curve(y_i, p_i)
            #print('\n step 03.02 ROC \n')
            roc_auc[i] = auc(fpr[i], tpr[i])
            #print('\n step 03.03 ROC \n')

        #print(fpr)
        #print(tpr)
        #print(roc_auc)

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        lw = 1

        font = {'family' : 'normal', 'weight' : 'normal', 'size': 8}
        plt.rc('font', **font)

        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ({0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ({0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=3)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0} ({1:0.2f})'
                     ''.format(names[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC OF Multi-Class')
        plt.legend(loc="lower right")
        plt.show()
        #plt.savefig('plot.png', dpi=300, bbox_inches='tight')

        return
    def on_train_end(self, logs={}):
        #print(self._data)
        names = self._names

        print("on_train_end")

    def get_data(self):
        return self._data