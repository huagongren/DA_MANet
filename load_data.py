"""
这是一个工具包，包含了获取各种数据集的方法
"""
from torch.utils.data import TensorDataset

from util import *
import pandas as pd

# 解析命令行参数
args = parse_arguments()
from imblearn.over_sampling import SMOTE, ADASYN

def get_daqing_multiscale(path):
    """
    获取大庆数据集的多尺度训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取训练数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)

    # 数据清洗，删除数据中包含任意缺失值的行
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    data_frame = data_frame[data_frame['Face'] != 5]
    # 选择要处理的列
    data_frame = data_frame.loc[:,
                 ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]

    # 分离标签得到特征
    X = data_frame.drop(labels='Face', axis=1)
    X = X.values
    # 对特征标准化
    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(X)

    # 取得标签
    y = data_frame['Face']
    y = y.values
    y = torch.LongTensor(y)

    # 获得多尺度数据,
    train_test_data = generate_multiscale_data(X, y)

    # 这里为什么y_train和y_test是是取得的train_test_data[2][0]和train_test_data[3][0]看generate_multiscale_data中的注释
    # 分离训练集和测试集的特征和标签
    X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2][0], \
    train_test_data[3][0]

    # 将多尺度数据转化为自定义的多尺度数据集，方便构造数据加载器
    train_dataset = MultiScaleDataset(X_train, y_train)
    test_dataset = MultiScaleDataset(X_test, y_test)

    # 构造数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return X_train, y_train, train_dataloader, X_test, y_test, test_dataloader


def handle_oversampling(X_train, y_train, high_target_classes, features, L):
    """
    对X_train中标签为high_target_classes的数据进行动态倍数的ADASYN过采样

    参数:
    X_train (list of torch.Tensor): 包含三个特征张量的列表，形状分别为：
        torch.Size([n_samples, 1, features]),
        torch.Size([n_samples, 3, features]),
        torch.Size([n_samples, 5, features])
    y_train (torch.Tensor): 标签张量，形状为[n_samples]
    high_target_classes (list): 需要过采样的目标类别列表
    L (list): 过采样级别列表，长度必须与high_target_classes一致，元素取值范围[0,5]
    features (int): 特征维度数（用于张量重塑）

    返回:
    X_updated (list of torch.Tensor): 过采样后的特征张量列表
    y_updated (torch.Tensor): 过采样后的标签张量
    """
    # 输入校验
    assert len(high_target_classes) == len(L), "high_target_classes和L的长度必须一致"

    # 展平特征张量并合并
    X_np_list = [xi.numpy().reshape(xi.shape[0], -1) for xi in X_train]
    X_combined = np.hstack(X_np_list)
    y_np = y_train.numpy()

    # 获取各类别原始样本数
    unique_classes, class_counts = np.unique(y_np, return_counts=True)
    class_count_dict = dict(zip(unique_classes, class_counts))

    # 动态生成过采样策略
    sampling_strategy = {}
    for cls, k in zip(high_target_classes, L):
        if cls in class_count_dict:
            original_count = class_count_dict[cls]
            multiplier = 1 + 0.5 * k  # 倍数计算逻辑
            target_count = int(original_count * multiplier)
            sampling_strategy[cls] = target_count

    # 执行ADASYN过采样
    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        random_state=42,
        n_neighbors=5
    )
    X_adasyn, y_adasyn = adasyn.fit_resample(X_combined, y_np)

    # 拆分并重塑张量
    split_sizes = [1 * features, 3 * features, 5 * features]
    X_adasyn_list = []
    start = 0
    for size in split_sizes:
        end = start + size
        c = size // features  # 计算原始通道数
        reshaped = torch.from_numpy(X_adasyn[:, start:end]).reshape(-1, c, features)
        X_adasyn_list.append(reshaped)
        start = end

    return X_adasyn_list, torch.from_numpy(y_adasyn)

# def random_scale_features(data, min_scale=0.8, max_scale=1.6, features = 11):
#     """
#     对具有features个特征的数据随机缩放其中几列。
#
#     参数:
#     data (numpy.ndarray): 输入的二维数组，每行代表一个样本，每列代表一个特征，特征数量应为features。
#     min_scale (float): 缩放因子的最小值，默认为0.5。
#     max_scale (float): 缩放因子的最大值，默认为2.0。
#
#     返回:
#     numpy.ndarray: 缩放后的二维数组。
#     """
#     # # 检查输入数据的特征数量是否为11
#     # if data.shape[1] != 11:
#     #     raise ValueError("输入数据的特征数量必须为11。")
#
#     # 随机选择要缩放的列
#     num_columns_to_scale = np.random.randint(1, features)  # 随机选择1到10列进行缩放
#     columns_to_scale = np.random.choice(features, num_columns_to_scale, replace=False)
#
#     # 为每个要缩放的列生成随机缩放因子
#     scaling_factors = np.random.uniform(min_scale, max_scale, num_columns_to_scale)
#
#     # 对选中的列进行缩放
#     scaled_data = data.copy()
#     for i, col in enumerate(columns_to_scale):
#         scaled_data[:, col] *= scaling_factors[i]
#
#     return scaled_data
def random_scale_features(data, k_values, features=11):
    """
    根据每个样本的k值动态调整缩放强度

    参数:
    data (numpy.ndarray): 输入的二维数组
    k_values (numpy.ndarray): 每个样本对应的缩放强度系数
    features (int): 特征维度数

    返回:
    numpy.ndarray: 缩放后的数据
    """
    scaled_data = data.copy()
    num_samples = data.shape[0]

    for i in range(num_samples):
        # 动态计算缩放范围
        k = k_values[i]
        center = 1.0

        # 当 k 为 0 时，不进行缩放操作
        if k == 0:
            continue

        # 随机选择并缩放特征列
        num_cols = np.random.randint(1, k + 1)
        cols = np.random.choice(features, num_cols, replace=False)
        factors = np.random.uniform(center - 0.1 * k, center + 0.1 * k, num_cols)
        scaled_data[i, cols] *= factors

    return scaled_data


import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import sklearn.preprocessing as preprocessing
# 需确保以下自定义函数/类已定义：
# random_scale_features, generate_multiscale_data, handle_oversampling, MultiScaleDataset, args（全局参数）

def get_daqing_multiscale_cuda(path, high_target_classes, low_target_classes, features, L_low, L_high):
    """
    获取大庆数据集的多尺度训练集和测试集（整合盲井实验的动态缩放+过采样逻辑）

    参数：
    - path: Excel文件的路径（str）
    - high_target_classes: 需要动态缩放的目标类别列表（list）
    - low_target_classes: 需要过采样的目标类别列表（list）
    - features: 需要缩放/过采样的特征列列表（list）
    - L_low: 过采样强度映射字典（key=类别, value=强度值）
    - L_high: 动态缩放强度映射字典（key=类别, value=强度值）

    返回：
    - X_train: 训练集特征（list，多尺度）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（list，多尺度）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """
    # ===================== 1. 数据加载与清洗（复用盲井逻辑） =====================
    data_frame = pd.read_excel(path)
    # 数据清洗：删除缺失值 + 剔除Face=5（凝灰岩）
    data_frame.dropna(inplace=True)
    data_frame = data_frame[data_frame['Face'] != 5]

    # ===================== 2. 特征工程 + 标准化（保留大庆原有逻辑） =====================
    # 选择特征列+标签列
    data_frame = data_frame.loc[:,
              ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离特征和标签
    X = data_frame.drop(labels='Face', axis=1).values
    y = data_frame['Face'].values
    # 特征标准化（大庆原有逻辑，盲井未提及，保留）
    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(X)

    # ===================== 3. 动态缩放（复用盲井的L_high映射逻辑） =====================
    X_res_1 = X.copy()
    y_res_1 = y.copy()

    # 生成高类别缩放强度列表（和盲井逻辑对齐）
    List_high = []
    for i in high_target_classes:
        List_high.append(L_high[i])
    print("高类别缩放强度 L_high：", L_high)

    # 对高目标类别执行动态缩放
    if high_target_classes:
        # 提取高目标类别的索引
        high_indices = np.isin(y_res_1, high_target_classes)
        # 生成类别→强度的映射表
        y_subset = y_res_1[high_indices]
        k_mapping = dict(zip(high_target_classes, List_high))
        k_values = np.array([k_mapping[cls] for cls in y_subset])
        # 执行动态缩放
        X_high_scaled = random_scale_features(
            X_res_1[high_indices],
            k_values=k_values,  # 传入动态强度值
            features=features
        )
        # 替换缩放后的数据
        X_res_1[high_indices] = X_high_scaled
    else:
        X_res_1, y_res_1 = X.copy(), y.copy()

    # ===================== 4. 生成多尺度数据（保留大庆原有划分逻辑） =====================
    # 常规实验：随机划分训练/测试集（而非盲井的按井划分）
    train_test_data = generate_multiscale_data(X_res_1, y_res_1)
    # 分离训练集/测试集的特征和标签
    X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2][0], train_test_data[3][0]

    # ===================== 5. 过采样（复用盲井的L_low映射逻辑） =====================
    # 生成低类别过采样强度列表（和盲井逻辑对齐）
    List_low = []
    for i in low_target_classes:
        List_low.append(L_low[i])
    print("低类别过采样强度 L_low：", L_low)

    # 核心修改：if 0 恒假，跳过过采样（和盲井逻辑对齐）
    if 0:  # 改为0，彻底关闭过采样
        X_res, y_res = handle_oversampling(X_train, y_train, low_target_classes, features, List_low)
    else:
        X_res, y_res = X_train, y_train  # 直接返回原数据，不执行过采样

    # ===================== 6. 构建数据集和数据加载器（复用原有逻辑） =====================
    train_dataset = MultiScaleDataset(X_res, y_res)
    test_dataset = MultiScaleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return X_train, y_train, train_dataloader, X_test, y_test, test_dataloader

def get_Hugoton_Panoma_multiscale(path):
    """
    获取Hugoton_Panoma数据集的多尺度训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件并获取新疆地区的训练数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)

    #数据清洗，删除数据中包含任意缺失值的行
    data_frame.dropna(inplace=True)
    #选择要处理的列
    data_frame = data_frame.loc[:,
              ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]

    # 分离标签得到特征
    X = data_frame.drop(labels='Facies', axis=1)

    # 对特征标准化
    max_min = preprocessing.StandardScaler()
    X = max_min.fit_transform(X)

    # 取得标签
    y = data_frame['Facies']
    y = y.values - 1 # 标签从1开始，这里将其转换为0开始
    y = torch.LongTensor(y)

    # 获得多尺度数据,
    train_test_data = generate_multiscale_data(X, y)

    # 这里为什么y_train和y_test是是取得的train_test_data[2][0]和train_test_data[3][0]看generate_multiscale_data中的注释
    # 分离训练集和测试集的特征和标签
    X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2][0], \
    train_test_data[3][0]

    # 将多尺度数据转化为自定义的多尺度数据集，方便构造数据加载器
    train_dataset = MultiScaleDataset(X_train, y_train)
    test_dataset = MultiScaleDataset(X_test, y_test)
    
    # 构造数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return X_train, y_train, train_dataloader, X_test, y_test, test_dataloader


def get_Hugoton_Panoma_multiscale_cuda(path, high_target_classes, low_target_classes, features, L_low, L_high):
    """
    获取Hugoton_Panoma数据集的多尺度训练集和测试集（整合动态缩放+过采样逻辑，过采样默认关闭）

    参数：
    - path: Excel文件的路径（str）
    - high_target_classes: 需要动态缩放的目标类别列表（list）
    - low_target_classes: 需要过采样的目标类别列表（list）
    - features: 需要缩放/过采样的特征列列表（list）
    - L_low: 过采样强度映射字典（key=类别, value=强度值）
    - L_high: 动态缩放强度映射字典（key=类别, value=强度值）

    返回：
    - X_train: 训练集特征（list，多尺度）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（list，多尺度）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """
    # ===================== 1. 数据加载与清洗（保留Hugoton_Panoma原有逻辑） =====================
    data_frame = pd.read_excel(path)
    # 数据清洗：删除包含任意缺失值的行
    data_frame.dropna(inplace=True)
    # 选择Hugoton_Panoma特有特征列+标签列
    data_frame = data_frame.loc[:,
                 ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]

    # ===================== 2. 特征工程 + 标准化（保留原有逻辑，变量命名规范化） =====================
    # 分离特征和标签
    X = data_frame.drop(labels='Facies', axis=1).values  # 转为numpy数组
    y = data_frame['Facies'].values - 1  # 标签从1→0开始，保持原有逻辑（先转numpy，避免后续np.isin报错）

    # 特征标准化（原max_min重命名为standard_scaler，和大庆逻辑对齐）
    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(X)

    # ===================== 3. 动态缩放（对齐大庆的L_high映射逻辑） =====================
    X_res_1 = X.copy()
    y_res_1 = y.copy()

    # 生成高类别缩放强度列表（和大庆逻辑对齐）
    List_high = []
    for i in high_target_classes:
        List_high.append(L_high[i])
    print("高类别缩放强度 L_high：", L_high)
    print("高类别目标强度列表 List_high：", List_high)

    # 对高目标类别执行动态缩放（修正原Tensor类型问题）
    if high_target_classes:
        # 提取高目标类别的索引（y_res_1是numpy数组，可正常使用np.isin）
        high_indices = np.isin(y_res_1, high_target_classes)
        # 生成类别→强度的映射表
        y_subset = y_res_1[high_indices]
        k_mapping = dict(zip(high_target_classes, List_high))
        k_values = np.array([k_mapping[cls] for cls in y_subset])
        # 执行动态缩放（传入k_values，对齐大庆的random_scale_features调用逻辑）
        X_high_scaled = random_scale_features(
            X_res_1[high_indices],
            k_values=k_values,  # 传入动态强度值
            features=features
        )
        # 替换缩放后的数据（原vstack逻辑等价于直接替换，更高效）
        X_res_1[high_indices] = X_high_scaled
    else:
        X_res_1, y_res_1 = X.copy(), y.copy()

    # ===================== 4. 生成多尺度数据（保留原有逻辑） =====================
    train_test_data = generate_multiscale_data(X_res_1, y_res_1)
    # 分离训练集/测试集的特征和标签（保留原有索引逻辑）
    X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2][0], \
        train_test_data[3][0]

    # ===================== 5. 过采样（对齐大庆的L_low映射逻辑，保持if 0关闭） =====================
    # 生成低类别过采样强度列表（和大庆逻辑对齐）
    List_low = []
    for i in low_target_classes:
        List_low.append(L_low[i])

    # 核心：if 0 恒假，跳过过采样（和大庆逻辑对齐）
    if 0:  # 改为0，彻底关闭过采样
        X_res, y_res = handle_oversampling(X_train, y_train, low_target_classes, features, List_low)
    else:
        X_res, y_res = X_train, y_train  # 直接返回原数据，不执行过采样

    # ===================== 6. 构建数据集和数据加载器（保留原有逻辑） =====================
    # 标签转为LongTensor（移到此处，避免前期numpy操作报错）
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # 构建自定义多尺度数据集
    train_dataset = MultiScaleDataset(X_res, y_res)
    test_dataset = MultiScaleDataset(X_test, y_test)

    # 构造数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return X_train, y_train, train_dataloader, X_test, y_test, test_dataloader


def get_part_daqing_multiscale(path):
    """
    获取部分大庆数据集的多尺度训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取训练数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)

    # 数据清洗，删除数据中包含任意缺失值的行
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    #data_frame = data_frame[data_frame['Face'] != 5]
    # 选择要处理的列
    data_frame = data_frame.loc[:,
                 ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]]

    # 分离标签得到特征
    X = data_frame.drop(labels='Face', axis=1)
    X = X.values
    # 对特征标准化
    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(X)

    # 取得标签
    y = data_frame['Face']
    y = y.values
    y = torch.LongTensor(y)

    # 获得多尺度数据,
    train_test_data = generate_multiscale_data(X, y)

    # 这里为什么y_train和y_test是是取得的train_test_data[2][0]和train_test_data[3][0]看generate_multiscale_data中的注释
    # 分离训练集和测试集的特征和标签
    X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2][0], \
    train_test_data[3][0]

    # 将多尺度数据转化为自定义的多尺度数据集，方便构造数据加载器
    train_dataset = MultiScaleDataset(X_train, y_train)
    test_dataset = MultiScaleDataset(X_test, y_test)

    # 构造数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return X_train, y_train, train_dataloader, X_test, y_test, test_dataloader


def get_blind1_multiscale(path, blind_well1, blind_well2):
    """
    获取盲井1数据集的多尺度训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

    参数：
    - path: Excel文件的路径（str）
    - blind_well1: 第一个盲井的名称（str）
    - blind_well2: 第二个盲井的名称（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件并获取盲测试数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    data_frame = data_frame[data_frame['Face'] != 5]

    # 训练集：排除两个盲井的所有数据
    train_frame = data_frame[(data_frame['井'] != blind_well1) &
                               (data_frame['井'] != blind_well2)]
    # 盲测试集：仅使用第一个盲井
    blind_frame = data_frame[data_frame['井'] == blind_well1]

    # 特征工程（训练集）
    train_data = train_frame.loc[:,
              ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_train = train_data.drop(labels='Face', axis=1)
    X_train= X_train.values
    # 取得标签
    y_train =train_frame['Face']
    y_train = y_train.values

    # 特征工程（盲测集）
    blind_data = blind_frame.loc[:,
              ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_blind = blind_data.drop(labels='Face', axis=1)
    X_blind = X_blind.values

    # 取得标签
    y_blind =blind_frame['Face']
    y_blind = y_blind.values

    # 生成盲井多尺度数据集，generate_multiscale_blind返回的是张量数据
    X_train,_ = generate_multiscale_blind(X_train, y_train)
    X_blind,_ = generate_multiscale_blind(X_blind, y_blind)
    y_train = torch.LongTensor(y_train)
    y_blind = torch.LongTensor(y_blind)
    train_dataset= MultiScaleDataset(X_train,y_train)
    test_dataset = MultiScaleDataset(X_blind,y_blind)

    # 构建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    blind_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return X_train,y_train,train_dataloader,X_blind,y_blind,blind_dataloader

def get_blind1_multiscale_cuda(path, blind_well1, blind_well2, high_target_classes, low_target_classes, features, L_low, L_high):
    """
    获取盲井1数据集的多尺度训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

    参数：
    - path: Excel文件的路径（str）
    - blind_well1: 第一个盲井的名称（str）
    - blind_well2: 第二个盲井的名称（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件并获取盲测试数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    data_frame = data_frame[data_frame['Face'] != 5]

    # 训练集：排除两个盲井的所有数据
    train_frame = data_frame[(data_frame['井'] != blind_well1) &
                               (data_frame['井'] != blind_well2)]
    # 盲测试集：仅使用第一个盲井
    blind_frame = data_frame[data_frame['井'] == blind_well1]

    # 特征工程（训练集）
    train_data = train_frame.loc[:,
              ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_train = train_data.drop(labels='Face', axis=1)
    X_train= X_train.values
    # 取得标签
    y_train =train_frame['Face']
    y_train = y_train.values

    List_high = []
    for i in high_target_classes:
        List_high.append(L_high[i])
    # for i in high_target_classes:
    #     if L_high[i] <=0:
    #         List_high.append(0)
    #     else:
    #         List_high.append(L_high[i])
    print(L_high)
    print(List_high)

    if  high_target_classes:
        X_res_1 = X_train.copy()
        y_res_1 = y_train.copy()
        high_indices = np.isin(y_train, high_target_classes)

        # 生成k值映射表
        y_subset = y_train[high_indices]
        k_mapping = dict(zip(high_target_classes, List_high))
        k_values = np.array([k_mapping[cls] for cls in y_subset])  # 关键映射逻辑

        # 动态缩放
        X_high_scaled = random_scale_features(
            X_train[high_indices],
            k_values=k_values,
            features=features
        )

        X_res_1[high_indices] = X_high_scaled
    else:
        X_res_1, y_res_1 = X_train.copy(), y_train.copy()

    # 特征工程（盲测集）
    blind_data = blind_frame.loc[:,
              ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_blind = blind_data.drop(labels='Face', axis=1)
    X_blind = X_blind.values

    # 取得标签
    y_blind =blind_frame['Face']
    y_blind = y_blind.values

    # 生成盲井多尺度数据集，generate_multiscale_blind返回的是张量数据
    X_train,_ = generate_multiscale_blind(X_res_1, y_res_1)
    X_blind,_ = generate_multiscale_blind(X_blind, y_blind)
    y_train = torch.LongTensor(y_res_1)
    y_blind = torch.LongTensor(y_blind)

    List_low = []
    for i in low_target_classes:
        List_low.append(L_low[i])
        # if L_low[i] >=0:
        #     List_low.append(0)
        # else:
        #     List_low.append(L_low[i])
    print(L_low)

    print(List_low)

    if 0:  # 仅当指定目标类别时过采样
        X_res, y_res = handle_oversampling(X_train, y_train, low_target_classes, features, List_low)
    else:
        X_res, y_res = X_train, y_train

    train_dataset= MultiScaleDataset(X_res,y_res)
    test_dataset = MultiScaleDataset(X_blind,y_blind)

    # 构建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    blind_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    return X_train,y_train,train_dataloader,X_blind,y_blind,blind_dataloader
def get_blind2_multiscale(path, blind_well):
    """
    获取Hugoton Panoma盲井数据集的多尺度训练集和测试集，数据加载器

    参数：
    - path: Excel文件的路径（str）
    - blind_well: 盲井名称（str）
    - high_target_classes: 需要特征缩放的目标类别列表（List[int]）
    - low_target_classes: 需要过采样的目标类别列表（List[int]）
    - features: 需要缩放的特征索引列表（List[int]）
    - L_low: 过采样系数字典（dict）
    - L_high: 特征缩放系数字典（dict）

    返回：
    - X_train: 训练集特征（Tensor）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_blind: 盲测集特征（Tensor）
    - y_blind: 盲测集岩性标签（Tensor）
    - blind_dataloader: 盲测集数据加载器（DataLoader）
    """

    # 从Excel读取数据
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 按井名分割数据集
    train_frame = data_frame[data_frame['Well Name'] != blind_well]  # 假设井列名为'Well'
    blind_frame = data_frame[data_frame['Well Name'] == blind_well]

    # 训练集处理
    train_data = train_frame.loc[:, ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    X_train = train_data.drop('Facies', axis=1).values
    y_train = train_data['Facies'].values - 1  # 标签转0-based

    # 盲测集处理
    blind_data = blind_frame.loc[:, ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    X_blind = blind_data.drop('Facies', axis=1).values
    y_blind = blind_data['Facies'].values - 1

    # 标准化处理
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_blind_scaled = scaler.transform(X_blind)



    # 生成多尺度数据
    X_train_multiscale, y_train = generate_multiscale_blind(X_train_scaled, y_train)
    X_blind_multiscale, y_blind = generate_multiscale_blind(X_blind_scaled, y_blind)

    # 转换为Tensor
    y_train = torch.LongTensor(y_train)
    y_blind = torch.LongTensor(y_blind)


    X_res, y_res = X_train_multiscale, y_train

    # 构建数据集
    train_dataset = MultiScaleDataset(X_res, y_res)
    blind_dataset = MultiScaleDataset(X_blind_multiscale, y_blind)

    # 构造数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    blind_dataloader = DataLoader(blind_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return X_res, y_res, train_dataloader, X_blind_multiscale, y_blind, blind_dataloader

def get_blind2_multiscale_cuda(path, blind_well, high_target_classes, low_target_classes, features, L_low, L_high):
    """
    获取Hugoton Panoma盲井数据集的多尺度训练集和测试集，数据加载器

    参数：
    - path: Excel文件的路径（str）
    - blind_well: 盲井名称（str）
    - high_target_classes: 需要特征缩放的目标类别列表（List[int]）
    - low_target_classes: 需要过采样的目标类别列表（List[int]）
    - features: 需要缩放的特征索引列表（List[int]）
    - L_low: 过采样系数字典（dict）
    - L_high: 特征缩放系数字典（dict）

    返回：
    - X_train: 训练集特征（Tensor）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_blind: 盲测集特征（Tensor）
    - y_blind: 盲测集岩性标签（Tensor）
    - blind_dataloader: 盲测集数据加载器（DataLoader）
    """

    # 从Excel读取数据
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 按井名分割数据集
    train_frame = data_frame[data_frame['Well Name'] != blind_well]  # 假设井列名为'Well'
    blind_frame = data_frame[data_frame['Well Name'] == blind_well]

    # 训练集处理
    train_data = train_frame.loc[:, ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    X_train = train_data.drop('Facies', axis=1).values
    y_train = train_data['Facies'].values - 1  # 标签转0-based

    # 盲测集处理
    blind_data = blind_frame.loc[:, ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    X_blind = blind_data.drop('Facies', axis=1).values
    y_blind = blind_data['Facies'].values - 1

    # 标准化处理
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_blind_scaled = scaler.transform(X_blind)

    # 高目标类别特征缩放
    if high_target_classes:
        List_high = [L_high[cls] for cls in high_target_classes]
        high_indices = np.isin(y_train, high_target_classes)

        # 动态缩放逻辑
        y_subset = y_train[high_indices]
        k_mapping = dict(zip(high_target_classes, List_high))
        k_values = np.array([k_mapping[cls] for cls in y_subset])

        X_high_scaled = random_scale_features(
            X_train_scaled[high_indices],
            k_values=k_values,
            features=features
        )
        X_train_scaled[high_indices] = X_high_scaled

    # 生成多尺度数据
    X_train_multiscale, y_train = generate_multiscale_blind(X_train_scaled, y_train)
    X_blind_multiscale, y_blind = generate_multiscale_blind(X_blind_scaled, y_blind)

    # 转换为Tensor
    y_train = torch.LongTensor(y_train)
    y_blind = torch.LongTensor(y_blind)

    # 低目标类别过采样
    if 0:
        List_low = [L_low[cls] for cls in low_target_classes]
        X_res, y_res = handle_oversampling(X_train_multiscale, y_train, low_target_classes, features, List_low)
    else:
        X_res, y_res = X_train_multiscale, y_train

    # 构建数据集
    train_dataset = MultiScaleDataset(X_res, y_res)
    blind_dataset = MultiScaleDataset(X_blind_multiscale, y_blind)

    # 构造数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    blind_dataloader = DataLoader(blind_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return X_res, y_res, train_dataloader, X_blind_multiscale, y_blind, blind_dataloader


def get_daqing(path):
    """
    获取大庆数据集的每个井的原始训练和测试数据（每个测井数据作为一个样本），同时生成数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - X_test: 测试集特征（numpy array）
    - train_loader: 训练集数据加载器（DataLoader）
    - y_train: 训练集岩性标签（Tensor）
    - y_test: 测试集岩性标签（Tensor）
    - test_loader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取大庆地区的原始训练和测试数据，同时生成数据加载器
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    data_frame = data_frame[data_frame['Face'] != 5]

    # 选择要处理的列
    data_frame = data_frame.loc[:,
              ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]

    # 分离标签得到特征
    X = data_frame.drop(labels='Face', axis=1)
    X = X.values
    # 对特征标准化
    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(X)

    # 取得标签
    y= data_frame['Face']
    y = y.values

    # 获得训练集和测试集的特征和标签,并转化为张量数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)

    # 构建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return X_train, X_test, train_loader, y_train, y_test, test_loader

def get_Hugoton_Panoma(path):
    """
    获取Hugoton-Panoma数据集的每个井的原始训练和测试数据（每个测井数据作为一个样本），同时生成数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - X_test: 测试集特征（numpy array）
    - train_loader: 训练集数据加载器（DataLoader）
    - y_train: 训练集岩性标签（Tensor）
    - y_test: 测试集岩性标签（Tensor）
    - test_loader: 测试集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件并获取新疆地区的原始训练和测试数据，同时生成数据加载器
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 选择要处理的列
    data_frame = data_frame.loc[:,
              ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]

    # 分离标签得到特征
    X = data_frame.drop(labels='Facies', axis=1)
    X = X.values
    # 对特征标准化
    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(X)

    # 取得标签
    y = data_frame['Facies']
    y = y.values - 1  # 关键修改：将标签值偏移-1，使得标签值从0开始

    # 获得训练集和测试集的特征和标签,并转化为张量数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)

    # 构建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return X_train, X_test, train_loader, y_train, y_test,test_loader

def get_part_daqing(path):
    """
    获取大庆数据集的每个井的原始训练和测试数据（每个测井数据作为一个样本），同时生成数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - X_test: 测试集特征（numpy array）
    - train_loader: 训练集数据加载器（DataLoader）
    - y_train: 训练集岩性标签（Tensor）
    - y_test: 测试集岩性标签（Tensor）
    - test_loader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取大庆地区的原始训练和测试数据，同时生成数据加载器
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    #data_frame = data_frame[data_frame['Face'] != 5]

    # 选择要处理的列
    data_frame = data_frame.loc[:,
              ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]]

    # 分离标签得到特征
    X = data_frame.drop(labels='Face', axis=1)
    X = X.values
    # 对特征标准化
    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(X)

    # 取得标签
    y= data_frame['Face']
    y = y.values

    # 获得训练集和测试集的特征和标签,并转化为张量数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)

    # 构建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return X_train, X_test, train_loader, y_train, y_test, test_loader


def get_blind1(path, blind_well1, blind_well2):
    """
    获取盲井1数据集的训练集和测试集（一个测井数据作为一个样本），数据加载器

    参数：
    - path: Excel文件的路径（str）
    - blind_well1: 第一个盲井的名称（str）
    - blind_well2: 第二个盲井的名称（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """


    # 从给定路径读取Excel文件并获取盲测试的原始训练和测试数据，同时生成数据加载器
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    data_frame = data_frame[data_frame['Face'] != 5]

    # 训练集：排除两个盲井的所有数据
    train_frame = data_frame[(data_frame['井'] != blind_well1) &
                             (data_frame['井'] != blind_well2)]
    # 盲测试集：仅使用第一个盲井
    blind_frame = data_frame[data_frame['井'] == blind_well1 | data_frame['井'] == blind_well2]

    # 特征工程（训练集）
    train_data = train_frame.loc[:,
                 ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_train = train_data.drop(labels='Face', axis=1)
    X_train = X_train.values
    # 取得标签
    y_train = train_frame['Face']
    y_train = y_train.values

    # 特征工程（盲测集）
    blind_data = blind_frame.loc[:,
                 ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_blind = blind_data.drop(labels='Face', axis=1)
    X_blind = X_blind.values
    # 取得标签
    y_blind = blind_frame['Face']
    y_blind = y_blind.values

    # 将训练集和盲测集的特征和标签转换为张量数据
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_blind = torch.tensor(X_blind)
    y_blind = torch.tensor(y_blind)

    # 构建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    blind_dataset = TensorDataset(X_blind, y_blind)
    blind_loader = DataLoader(blind_dataset, batch_size=args.batch_size, shuffle=True)

    return X_train, X_blind, train_loader, y_train, y_blind, blind_loader

def get_blind2(path, blind_well1, blind_well2):
    """
    获取盲井1数据集的训练集和测试集（一个测井数据作为一个样本），数据加载器

    参数：
    - path: Excel文件的路径（str）
    - blind_well1: 第一个盲井的名称（str）
    - blind_well2: 第二个盲井的名称（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取盲测试的原始训练和测试数据，同时生成数据加载器
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    data_frame = data_frame[data_frame['Face'] != 5]

    # 训练集：排除两个盲井的所有数据
    train_frame = data_frame[(data_frame['井'] != blind_well1) &
                             (data_frame['井'] != blind_well2)]
    # 盲测试集：仅使用第二个盲井
    blind_frame = data_frame[data_frame['井'] == blind_well2]

    # 特征工程（训练集）
    train_data = train_frame.loc[:,
                 ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_train = train_data.drop(labels='Face', axis=1)
    X_train = X_train.values
    # 取得标签
    y_train = train_frame['Face']
    y_train = y_train.values

    # 特征工程（盲测集）
    blind_data = blind_frame.loc[:,
                 ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_blind = blind_data.drop(labels='Face', axis=1)
    X_blind = X_blind.values
    # 取得标签
    y_blind = blind_frame['Face']
    y_blind = y_blind.values

    # 将训练集和盲测集的特征和标签转换为张量数据
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_blind = torch.tensor(X_blind)
    y_blind = torch.tensor(y_blind)

    # 构建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(X_blind, y_blind)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return X_train, X_blind, train_loader, y_train, y_blind, test_loader



def get_daqing_time_series(path):
    """
    读取大庆测井数据，进行数据清洗、标准化处理，并转换为时间序列数据格式，返回训练集和测试集的数据加载器。

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（Tensor）
    - y_train: 训练集岩性标签（Tensor）
    - train_loader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（Tensor）
    - y_test: 测试集岩性标签（Tensor）
    - test_loader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取训练数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)
    # 数据清洗，删除数据中包含任意缺失值的行
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    data_frame = data_frame[data_frame['Face'] != 5]
    # 选择要处理的列
    data_frame = data_frame.loc[:,
              ["井","AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]

    # 特征工程
    features = data_frame.drop('Face', axis=1)
    labels = data_frame['Face']

    # 数据标准化（应在分割后进行以避免数据泄漏）
    # 对于时间序列数据的分割方式有可能需要改进，既要保证随机性保证数据集中的时序关系不被破坏，目前采用随机划分的方式
    X_train, X_test, y_train, y_test = split_by_well1(
        features,
        labels,
        "井",
        test_size=args.test_size,
        random_state=42,
        shuffle=True
    )
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # 创建训练集和测试集的时间序列数据集
    train_dataset = TimeSeriesDataset(X_train, y_train, args.time_step_size)
    test_dataset = TimeSeriesDataset(X_test, y_test, args.time_step_size)
    # 将X_train和y_train转换为时间序列数据格式
    X_train,y_train = get_time_series_dataset(X_train,y_train, args.time_step_size)
    # 将X_test和y_test转换为时间序列数据格式
    X_test,y_test = get_time_series_dataset(X_test,y_test, args.time_step_size)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return X_train,y_train , train_loader, X_test, y_test, test_loader


def get_Hugoton_Panoma_time_series(path):
    """
    读取Hugoton_Panoma测井数据，进行数据清洗、标准化处理，并转换为时间序列数据格式，返回训练集和测试集的数据加载器。

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（Tensor）
    - y_train: 训练集岩性标签（Tensor）
    - train_loader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（Tensor）
    - y_test: 测试集岩性标签（Tensor）
    - test_loader: 测试集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件并获取训练数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)
    # 数据清洗，删除数据中包含任意缺失值的行
    data_frame.dropna(inplace=True)
    # 选择特征
    data_frame = data_frame.loc[:,
              ["Well Name", "GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    # 特征工程
    features = data_frame.drop('Facies', axis=1)
    labels = data_frame['Facies']

    # 数据标准化（应在分割后进行以避免数据泄漏）
    # 对于时间序列数据的分割方式有可能需要改进，既要保证随机性保证数据集中的时序关系不被破坏，目前采用随机划分的方式
    X_train, X_test, y_train, y_test = split_by_well1(
        features,
        labels,
        "Well Name",
        test_size=args.test_size,
        random_state=42,
        shuffle=True
    )
    # 新增：让标签从0开始（关键修改）
    y_train = y_train - 1  # 训练集标签减1
    y_test = y_test - 1  # 测试集标签减1
    assert y_train.min() >= 0, "训练集标签存在负数"
    assert y_test.min() >= 0, "测试集标签存在负数"

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # 创建训练集和测试集的时间序列数据集
    train_dataset = TimeSeriesDataset(X_train, y_train, args.time_step_size)
    test_dataset = TimeSeriesDataset(X_test, y_test, args.time_step_size)
    # 将X_train和y_train转换为时间序列数据格式
    X_train, y_train = get_time_series_dataset(X_train, y_train, args.time_step_size)
    # 将X_test和y_test转换为时间序列数据格式
    X_test, y_test = get_time_series_dataset(X_test, y_test, args.time_step_size)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return X_train, y_train, train_loader, X_test, y_test, test_loader

def get_part_daqing_time_series(path):
    """
    读取大庆测井数据，进行数据清洗、标准化处理，并转换为时间序列数据格式，返回训练集和测试集的数据加载器。

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（Tensor）
    - y_train: 训练集岩性标签（Tensor）
    - train_loader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（Tensor）
    - y_test: 测试集岩性标签（Tensor）
    - test_loader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取训练数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)
    # 数据清洗，删除数据中包含任意缺失值的行
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    #data_frame = data_frame[data_frame['Face'] != 5]
    # 选择要处理的列
    data_frame = data_frame.loc[:,
                 ["井", "SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish",
                  "Face"]]

    # 特征工程
    features = data_frame.drop('Face', axis=1)
    labels = data_frame['Face']

    # 数据标准化（应在分割后进行以避免数据泄漏）
    # 对于时间序列数据的分割方式有可能需要改进，既要保证随机性保证数据集中的时序关系不被破坏，目前采用随机划分的方式
    X_train, X_test, y_train, y_test = train_test_split(
        features.values,
        labels.values,
        "井",
        test_size=args.test_size,
        random_state=42,
        shuffle=True
    )
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # 创建训练集和测试集的时间序列数据集
    train_dataset = TimeSeriesDataset(X_train, y_train, args.time_step_size)
    test_dataset = TimeSeriesDataset(X_test, y_test, args.time_step_size)
    # 将X_train和y_train转换为时间序列数据格式
    X_train,y_train = get_time_series_dataset(X_train,y_train, args.time_step_size)
    # 将X_test和y_test转换为时间序列数据格式
    X_test,y_test = get_time_series_dataset(X_test,y_test, args.time_step_size)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return X_train,y_train , train_loader, X_test, y_test, test_loader


def get_blind1_time_series(path, blind_well1, blind_well2):
    """
    获取盲井1的时间序列数据的训练集和盲测集，并转换为时间序列数据格式，返回数据加载器。

    参数：
    - path: Excel文件的路径（str）
    - blind_well1: 第一个盲井的名称（str）
    - blind_well2: 第二个盲井的名称（str）

    返回：
    - X_train: 时间序列数据的训练集特征（Tensor）
    - X_blind: 时间序列数据的盲测集特征（Tensor）
    - train_loader: 时间序列数据的训练集数据加载器（DataLoader）
    - y_train: 时间序列数据的训练集岩性标签（Tensor）
    - y_blind: 时间序列数据的盲测集岩性标签（Tensor）
    - blind_dataset_loader: 时间序列数据的盲测集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件并获取盲测试数据
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    data_frame = data_frame[data_frame['Face'] != 5]

    # 训练集：排除两个盲井的所有数据
    train_frame = data_frame[(data_frame['井'] != blind_well1) &
                             (data_frame['井'] != blind_well2)]
    # 盲测试集：仅使用第一个盲井
    blind_frame = data_frame[data_frame['井'] == blind_well1]

    #特征工程（训练集）
    train_data = train_frame.loc[:,
                 ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_train = train_data.drop(labels='Face', axis=1)
    X_train = X_train.values
    # 取得标签
    y_train = train_frame['Face']
    y_train = y_train.values

    # 特征工程（盲测集）
    blind_data = blind_frame.loc[:,
                 ["AC", "At10", "At20", "At30", "At60", "At90", "CNL", "DEN", "GR", "PE", "SP", "Face"]]
    # 分离标签得到特征
    X_blind = blind_data.drop(labels='Face', axis=1)
    X_blind = X_blind.values
    # 取得标签
    y_blind = blind_frame['Face']
    y_blind = y_blind.values

    # 生成盲井时间序列数据集
    train_dataset = TimeSeriesDataset(X_train, y_train, args.time_step_size)
    blind_dataset = TimeSeriesDataset(X_blind, y_blind, args.time_step_size)
    # 将X_train和y_train转换为时间序列数据格式
    X_train, y_train = get_time_series_dataset(X_train, y_train, args.time_step_size)
    # 将X_test和y_test转换为时间序列数据格式
    X_blind, y_blind = get_time_series_dataset(X_blind, y_blind, args.time_step_size)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    blind_dataset_loader = DataLoader(blind_dataset, batch_size=args.batch_size, shuffle=True)
    return X_train, X_blind, train_loader, y_train, y_blind, blind_dataset_loader


def get_blind2_time_series(path, blind_well):
    """
    获取Hugoton_Panoma数据集中单个盲井的时间序列数据，返回训练集和盲测集。

    参数：
    - path: Excel文件的路径（str）
    - blind_well: 盲井的名称（str）

    返回：
    - X_train: 时间序列数据的训练集特征（Tensor）
    - X_blind: 时间序列数据的盲测集特征（Tensor）
    - train_loader: 时间序列数据的训练集数据加载器（DataLoader）
    - y_train: 时间序列数据的训练集岩性标签（Tensor）
    - y_blind: 时间序列数据的盲测集岩性标签（Tensor）
    - blind_dataset_loader: 时间序列数据的盲测集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 选择与Hugoton_Panoma数据一致的特征
    data_frame = data_frame.loc[:,
                 ["Well Name", "GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]

    # 训练集：排除盲井的所有数据
    train_frame = data_frame[data_frame['Well Name'] != blind_well]
    # 盲测试集：仅使用指定的盲井
    blind_frame = data_frame[data_frame['Well Name'] == blind_well]

    # 特征工程（训练集）
    train_data = train_frame.loc[:,
                 ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    # 分离标签得到特征
    X_train = train_data.drop(labels='Facies', axis=1)
    y_train = train_data['Facies']

    # 特征工程（盲测集）
    blind_data = blind_frame.loc[:,
                 ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    # 分离标签得到特征
    X_blind = blind_data.drop(labels='Facies', axis=1)
    y_blind = blind_data['Facies']

    # 转换为numpy数组
    X_train = X_train.values
    y_train = y_train.values
    X_blind = X_blind.values
    y_blind = y_blind.values

    # 重要：让标签从0开始（Hugoton_Panoma数据集中岩性标签是1-9，减1后变为0-8）
    y_train = y_train - 1
    y_blind = y_blind - 1

    # 检查标签范围
    print(f"训练集标签范围: [{y_train.min()}, {y_train.max()}]")
    print(f"盲测集标签范围: [{y_blind.min()}, {y_blind.max()}]")
    print(f"训练集唯一标签: {np.unique(y_train)}")
    print(f"盲测集唯一标签: {np.unique(y_blind)}")

    # 数据标准化（应在分割后进行以避免数据泄漏）
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_blind = scaler.transform(X_blind)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_blind = torch.FloatTensor(X_blind)
    y_train = torch.LongTensor(y_train)
    y_blind = torch.LongTensor(y_blind)

    # 创建时间序列数据集
    train_dataset = TimeSeriesDataset(X_train, y_train, args.time_step_size)
    blind_dataset = TimeSeriesDataset(X_blind, y_blind, args.time_step_size)

    # 将X_train和y_train转换为时间序列数据格式
    X_train_ts, y_train_ts = get_time_series_dataset(X_train, y_train, args.time_step_size)
    # 将X_blind和y_blind转换为时间序列数据格式
    X_blind_ts, y_blind_ts = get_time_series_dataset(X_blind, y_blind, args.time_step_size)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    blind_dataset_loader = DataLoader(blind_dataset, batch_size=args.batch_size, shuffle=False)  # 注意：测试时通常不shuffle

    print(f"训练集大小: {len(train_dataset)} 个样本")
    print(f"盲测集大小: {len(blind_dataset)} 个样本")

    return X_train_ts, X_blind_ts, train_loader, y_train_ts, y_blind_ts, blind_dataset_loader
