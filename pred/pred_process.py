import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os


def get_tc_len(path='D:/pycharm/EnsTyT/CMA_dataset/CMA_csv/CMA_tc_all.csv', split_years=[1979, 2016, 2021], is_save=True,
               train_tc_num_path='./processed/train_tc_num.npy', test_tc_num_path='./processed/test_tc_num.npy',
               train_tc_name_path='./processed/train_tc_name.npy', test_tc_name_path='./processed/test_tc_name.npy'):
    if not os.path.exists(path):
        raise FileExistsError('预处理的台风数据文件' + path + '不存在')
    csv = pd.read_csv(path, index_col=0)
    tc = csv['storm_id'].unique()
    # 获取训练集和测试集中每个台风的样本数量
    print('处理台风样本数信息：')
    tc_name = []
    n = []
    check = True
    for t in tqdm(tc):
        if str(t)[:4] == str(split_years[1]) and check:
            train_storm_num = n
            n = []
            check = False
        r = csv[csv['storm_id'] == t].reset_index()
        if len(r) < 9:
            continue
        else:
            n.append(len(range(4, len(r) - 4)))
            tc_name.append(str(r['name'].unique()[0]))
    test_storm_num = n
    train_storm_num, test_storm_num = np.array(train_storm_num), np.array(test_storm_num)
    train_tc_name, test_tc_name = tc_name[:len(train_storm_num)], tc_name[len(train_storm_num):]
    train_tc_name, test_tc_name = np.array(train_tc_name, dtype=str), np.array(test_tc_name, dtype=str)
    if is_save:
        np.save(train_tc_num_path, train_storm_num)
        np.save(test_tc_num_path, test_storm_num)
        np.save(train_tc_name_path, train_tc_name)
        np.save(test_tc_name_path, test_tc_name)
    return train_storm_num, test_storm_num, train_tc_name, test_tc_name


def pred_process(train_path='./train/', test_path='./test/'):
    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)
    train_files = sorted(train_files, key=lambda x: int(x[5:8]))
    test_files = sorted(test_files, key=lambda x: int(x[5:8]))
    _, _, _, _ = get_tc_len(is_save=True)

    # 训练集数据合并
    x_train = []
    a = np.load(train_path + train_files[0], allow_pickle=True)
    y_train = a[:, :, 2:]

    print('处理训练集数据：')
    for f in tqdm(train_files):
        a = np.load(train_path + f, allow_pickle=True)
        x_train.append(np.expand_dims(a[:, :, :2], axis=1))
    x_train = np.concatenate(x_train, axis=1)
    print('x_train shape=', x_train.shape)
    print('y_train shape=', y_train.shape)
    time.sleep(0.1)

    # 测试集数据合并
    x_test = []
    a = np.load(test_path + test_files[0], allow_pickle=True)
    y_test = a[:, :, 2:]

    print('处理测试集数据：')
    for f in tqdm(test_files):
        a = np.load(test_path + f, allow_pickle=True)
        x_test.append(np.expand_dims(a[:, :, :2], axis=1))
    x_test = np.concatenate(x_test, axis=1)
    print('x_test shape=', x_test.shape)
    print('y_test shape=', y_test.shape)

    # 保存数据
    np.save('./processed/x_train.npy', x_train)
    np.save('./processed/y_train.npy', y_train)
    np.save('./processed/x_test.npy', x_test)
    np.save('./processed/y_test.npy', y_test)

def pred_process_2021_now(test_path='./test_2021_now/'):
    test_files = os.listdir(test_path)
    test_files = sorted(test_files, key=lambda x: int(x[5:8]))
    _, draw_tc_num, _, draw_tc_name = get_tc_len(split_years=[1979, 2021, 2024], is_save=False)
    np.save('./processed/draw_tc_num.npy', draw_tc_num)
    np.save('./processed/draw_tc_name.npy', draw_tc_name)
    # 数据合并
    x_test = []
    a = np.load(test_path + test_files[0], allow_pickle=True)
    y_test = a[:, :, 2:]
    print('处理测试集数据：')
    for f in tqdm(test_files):
        a = np.load(test_path + f, allow_pickle=True)
        x_test.append(np.expand_dims(a[:, :, :2], axis=1))
    x_test = np.concatenate(x_test, axis=1)
    print('x_test shape=', x_test.shape)
    print('y_test shape=', y_test.shape)
    # 保存数据
    np.save('./processed/x_ens_test.npy', x_test)
    np.save('./processed/y_ens_test.npy', y_test)

# pred_process()
pred_process_2021_now()
'''
_, b, _, c = get_tc_len(split_years=[1979, 2021, 2024], is_save=False)
print(c)
np.save('./processed/draw_tc_num.npy', b)
np.save('./processed/draw_tc_name.npy', c)
'''
