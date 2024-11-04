import math
import scipy.io as sio
import torch
import torch.utils.data as Data
import numpy as np
from sklearn.decomposition import PCA
def load_dataset(Dataset,K):
    path = './dataset/'
    if Dataset == 'IP':
        mat_data = sio.loadmat(path + 'Indian_pines_corrected.mat') #200bands
        mat_gt = sio.loadmat(path + 'Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        TOTAL_SIZEBG = 21025

    if Dataset == 'PU':
        uPavia = sio.loadmat(path + 'PaviaU.mat')  #103bands
        gt_uPavia = sio.loadmat(path + 'PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        TOTAL_SIZEBG = 207400

    if Dataset == 'SA':
        SV = sio.loadmat(path + 'Salinas_corrected.mat')#204bands
        gt_SV = sio.loadmat(path + 'Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        TOTAL_SIZEBG = 111104

    if Dataset == 'KSC':
        data_hsi = sio.loadmat(path + 'KSC.mat')['KSC']
        gt_hsi = sio.loadmat(path + 'KSC_gt.mat')['KSC_gt']
        TOTAL_SIZE = 0
        TOTAL_SIZEBG = 314368

    if Dataset == 'HHK':
        data_hsi = sio.loadmat(path + 'ZY_hhk.mat')['ZY_hhk_0628_data']
        gt_hsi = sio.loadmat(path + 'ZY_hhk_gt.mat')['data_gt']
        TOTAL_SIZE = 9825
        TOTAL_SIZEBG = 1835200

    if Dataset == 'HANCHUAN':
        data_hsi = sio.loadmat(path + 'WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
        gt_hsi = sio.loadmat(path + 'WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
        TOTAL_SIZE = 257530
        TOTAL_SIZEBG = 368751

    if Dataset == 'LONGKOU':
        data_hsi = sio.loadmat(path + 'WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        gt_hsi = sio.loadmat(path + 'WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
        TOTAL_SIZE = 204542
        TOTAL_SIZEBG = 220000

    if Dataset == 'HONGHU':
        data_hsi = sio.loadmat(path + 'WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
        gt_hsi = sio.loadmat(path + 'WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
        TOTAL_SIZE = 386693
        TOTAL_SIZEBG = 446500

    shapeor = data_hsi.shape
    data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
    data_hsi = PCA(n_components=K).fit_transform(data_hsi)
    shapeor = np.array(shapeor)
    shapeor[-1] = K
    data_hsi = data_hsi.reshape(shapeor)
    return data_hsi, gt_hsi, TOTAL_SIZE,TOTAL_SIZEBG
def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign
def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch
def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension),dtype=np.float32)
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data
def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                    PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                   PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    x_test = x_test_all
    y_test = y_test

    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    x1_tensor_train.shape
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)

    train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,)
    test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,)

    return train_iter, test_iter  # , y_test

def sampleDIY(num_perclass,ground_truth,bg=False):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth) + 1
    for i in range(m):
        a = 0 if bg == True else 1
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + a]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if num_perclass > int(len(indexes))//2:
            nb_val = int(len(indexes))//2
        else:
            nb_val = num_perclass
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes