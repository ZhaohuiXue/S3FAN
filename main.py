import time
import torch
from record import record_output, aa_and_each_accuracy
import numpy as np
from sklearn import metrics, preprocessing
from dataprocess import load_dataset,sampleDIY
from sklearn.metrics import confusion_matrix
from model import S3FAN
from dataprocess import generate_iter
import collections
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy

ITER = 10
patch = 7       #patch  IP7  SV、HHK15  PU13 LONGKOU 9
PATCH_LENGTH = patch // 2
lr = 0.001
epochs = 100
PC = 110                  #IP110 PU50 SA40 HHK90 LONGKOU40
KS = 10
dataset = 'IP'                #IP PU SA LONGKOU HHK
Dataset = dataset.upper()
MODEL = 'S3FAN'
batch = 32
num_perclass = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]

data_hsi, gt_hsi, TOTAL_SIZE,TOTAL_SIZEBG = load_dataset(Dataset,PC)
print(data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

loss = LabelSmoothingCrossEntropy(smoothing=0.1)
img_rows = str(2*PATCH_LENGTH+1)
img_channels = data_hsi.shape[2]
channels = data_hsi.shape[2]
num_classes = CLASSES_NUM
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))
data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
#########
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                        'constant', constant_values=0)
MIN_NUM_PATCHES = 16

def train(model_S3FAN, train_iter, loss, optimizer, device, epochs):
    print("training on ", device)
    train_loss_list = []
    train_acc_list = []
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80, eta_min=0.0, last_epoch=-1) #
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            label = y.to(torch.int64)
            result = model_S3FAN(X)
            l = loss(result,label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (result.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        lr_adjust.step(epoch)
        train_loss_list.append(train_l_sum)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)

        print('epoch %d, train loss %.6f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
                 time.time() - time_epoch))

for index_iter in range(ITER):
    print('iter:', index_iter)

    model_S3FAN = S3FAN(
    image_size=patch,
    num_classes=num_classes,
    KS=KS,
    dim=63,
    depth=3,
    heads=3,
    channels=channels,
    dropout=0.1,
    emb_dropout=0.1,
    dim_head=63//3,
    model = MODEL
    ).cuda()
    optimizer = torch.optim.Adam(model_S3FAN.parameters(), lr=lr, weight_decay=1e-8)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampleDIY(num_perclass, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = len(test_indices)

    print('-----Selecting Small Pieces from the Original Cube Data-----')
    train_iter, test_iter = \
    generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch, gt)

    tic1 = time.time()
    train(model_S3FAN, train_iter, loss, optimizer, device, epochs=epochs)
    toc1 = time.time()

    pred_test = []
    tic2 = time.time()
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            model_S3FAN.eval()
            y_hat = model_S3FAN(X)
            pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
    toc2 = time.time()
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1

    overall_acc = metrics.accuracy_score(gt_test[:],pred_test)
    confusion_matrix = metrics.confusion_matrix(gt_test[:],pred_test)
    each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(gt_test[:],pred_test)
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc
    print('OA:',overall_acc)

record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,confusion_matrix,
                     './content/' + Dataset + str(num_perclass) + '.txt')


