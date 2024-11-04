import numpy as np
from operator import truediv
def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, confusion_matrix, path):
    f = open(path, 'a')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(round(np.mean(oa_ae)*100, 2)) + ' ± ' + str(round(np.std(oa_ae)*100, 2)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(round(np.mean(aa_ae)*100, 2)) + ' ± ' + str(round(np.std(aa_ae)*100, 2)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(round(np.mean(kappa_ae)*100,2)) + ' ± ' + str(round(np.std(kappa_ae)*100,2)) + '\n' + '\n'
    f.write(sentence5)
    element_mean = np.mean(element_acc_ae*100, axis=0)
    element_std = np.std(element_acc_ae*100, axis=0)
    sentence6 = "Mean and std of all elements in confusion matrix: " + '\n'
    f.write(sentence6)
    for i in range(len(element_mean)):
        sentence = str(round(element_mean[i],2)) + ' ± ' + str(round(element_std[i],2)) + '\n'
        f.write(sentence)

    sentence8 = 'Total average Training time is: ' + str(np.sum(training_time_ae)) + '\n'
    f.write(sentence8)
    sentence9 = 'Total average Testing time is: ' + str(np.sum(testing_time_ae)) + '\n' + '\n'
    f.write(sentence9)
    sentence7 = "The diagonal Confusion matrix: " + '\n' + str(confusion_matrix) + '\n'
    f.write(sentence7)

    f.close()



