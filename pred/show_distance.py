import numpy as np
import os

def compute_distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = lat1.astype('float'), lon1.astype('float'), lat2.astype('float'), lon2.astype('float')
    # 批量计算地球上两点间的球面距离
    R = 6371e3  # 地球半径（米）
    phi_1, phi_2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.power(np.sin(delta_phi / 2), 2) + np.cos(phi_1) * np.cos(phi_2) * np.power(np.sin(delta_lambda / 2), 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c / 1000.


def compute_distance_of_models(train_folder='./train/', test_folder='./test/'):
    train_distance, test_distance, train_sd, test_sd = [], [], [], []
    train_files = os.listdir(train_folder)
    test_files = os.listdir(test_folder)
    train_files = sorted(train_files, key=lambda x: int(x[5:8]))
    test_files = sorted(test_files, key=lambda x: int(x[5:8]))

    for f1, f2 in zip(train_files, test_files):
        pred1 = np.load(train_folder + f1, allow_pickle=True)
        pred2 = np.load(test_folder + f2, allow_pickle=True)
        time_step1, time_step2, time_step1_v, time_step2_v = [], [], [], []
        for i in range(4):
            distance1 = compute_distance_km(pred1[:, i, 0], pred1[:, i, 1], pred1[:, i, 2], pred1[:, i, 3])
            distance2 = compute_distance_km(pred2[:, i, 0], pred2[:, i, 1], pred2[:, i, 2], pred2[:, i, 3])
            time_step1.append(distance1.mean())
            time_step2.append(distance2.mean())
            time_step1_v.append(np.std(distance1))
            time_step2_v.append(np.std(distance2))
        train_distance.append(time_step1)
        test_distance.append(time_step2)
        train_sd.append(time_step1_v)
        test_sd.append(time_step2_v)

    train_distance = np.array(train_distance)
    test_distance = np.array(test_distance)
    train_sd = np.array(train_sd)
    test_sd = np.array(test_sd)
    print('Avg Train Distance:', train_distance.mean(axis=0), 'Avg Test Distance:', test_distance.mean(axis=0))
    np.save('./processed/train_distance.npy', train_distance)
    np.save('./processed/test_distance.npy', test_distance)
    np.save('./processed/train_standard.npy', train_sd)
    np.save('./processed/test_standard.npy', test_sd)

def check_distance_of_saves(folder='./processed/'):
    p = np.load(folder + 'x_train.npy', allow_pickle=True)
    t = np.load(folder + 'y_train.npy', allow_pickle=True)
    s = np.load(folder + 'train_distance.npy', allow_pickle=True)
    # print(p.shape, t.shape)
    for i in range(p.shape[1]):
        d = []
        for j in range(4):
            d.append(compute_distance_km(p[:, i, j, 0], p[:, i, j, 1], t[:, j, 0], t[:, j, 1]).mean())
        d = np.array(d)
        print(d, s[i])


compute_distance_of_models()
