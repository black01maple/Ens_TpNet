import numpy as np
import os


def get_sort_index(train_distance_path='./processed/train_distance.npy'):
    a = np.load(train_distance_path, allow_pickle=True)
    c = np.linspace(0, len(a) - 1, len(a)).reshape(-1, 1)
    a = a[:, -1].reshape(-1, 1)
    a = np.concatenate([c, a], axis=1)
    a = list(a)
    a = sorted(a, key=lambda x: x[1])
    sort_index = []
    for i in a:
        sort_index.append(i[0])
    sort_index = np.array(sort_index, dtype=int)
    np.save('./processed/temp_sort_index', sort_index)
    return sort_index

def sort_pred(sort_index, pred_folder='./processed/'):
    a = np.load(pred_folder + 'x_train.npy', allow_pickle=True)
    a = a[:, sort_index]
    b = np.load(pred_folder + 'x_test.npy', allow_pickle=True)
    b = b[:, sort_index]
    c = np.load(pred_folder + 'train_distance.npy', allow_pickle=True)
    c = c[sort_index]
    d = np.load(pred_folder + 'test_distance.npy', allow_pickle=True)
    d = d[sort_index]
    e = np.load(pred_folder + 'train_standard.npy', allow_pickle=True)
    e = e[sort_index]
    f = np.load(pred_folder + 'test_standard.npy', allow_pickle=True)
    f = f[sort_index]
    np.save(pred_folder + 'x_train.npy', a)
    np.save(pred_folder + 'x_test.npy', b)
    np.save(pred_folder + 'train_distance.npy', c)
    np.save(pred_folder + 'test_distance.npy', d)
    np.save(pred_folder + 'train_standard.npy', e)
    np.save(pred_folder + 'test_standard.npy', f)

def rename_models(sort_index, model_save_folder='D:/pycharm/EnsTyT/model/'):
    names = os.listdir(model_save_folder)
    names = sorted(names, key=lambda x: int(x[5:8]))
    names = np.array(names)
    names_new = names.copy()
    names_new = names_new[sort_index]
    for i, j in zip(names, names_new):
        os.rename(model_save_folder + i, model_save_folder + '0' + j)
    for j in names_new:
        os.rename(model_save_folder + '0' + j, model_save_folder + j)


a=get_sort_index()
sort_pred(a)
# rename_models(a)