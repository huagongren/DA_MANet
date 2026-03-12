"""
这是一个工具包，包含了一些辅助函数，包括一些关键函数和类
"""
import argparse
import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.table import Table

def parse_arguments():
    """
    解析命令行参数，主要用于设置超参数。

    参数：
    无

    返回：
    - args: 解析后的参数对象，包含以下参数：
        - dataset: 使用的数据集（'daqing' 或 'Hugoton_Panoma'）。
        - no_cuda: 是否禁用CUDA（布尔值）。
        - seed: 随机种子（int）。
        - epochs: 训练的轮数（int）。
        - lr: 初始学习率（float）。
        - gamma: 学习率调度的衰减系数（float）。
        - step_size: 学习率调度的步长（int）。
        - data_path1: 大庆数据集路径（str）。
        - data_path2: Hugoton_Panoma数据集路径（str）。
        - dropout: Dropout概率（float）。
        - batch_size: 训练批次大小（int）。
        - test_size: 测试集占比（float）。
        - num_classes: 数据集中的类别数量（int）。
        - features1: 大庆数据集的特征数（int）。
        - features2: Hugoton_Panoma数据集的特征数（int）。
        - weights1: 数据集1的Focal Loss权重（list of float）。
        - weights2: 数据集2的Focal Loss权重（list of float）。
        - weights3: 数据集3的Focal Loss权重（list of float）。
        - hidden_size: BiLSTM隐藏层大小（int）。
        - num_layers: BiLSTM和BiGRU的层数（int）。
        - time_step_size: BiLSTM和BiGRU的时间步大小（int）。
        - blind_well1: 盲井1的名称（str）。
        - blind_well2: 盲井2的名称（str）。
        - cuda: 是否使用CUDA（布尔值）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='daqing', help='Dataset to use: daqing or Hugoton_Panoma or part_daqing or blind1 or blind2. ')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=244, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR scheduler gamma.')
    parser.add_argument('--step-size', type=int, default=20, help='LR scheduler step size.')
    parser.add_argument('--data_path1', type=str, default="dataset/all_Daqing.xlsx", help='daqing_path.')
    parser.add_argument('--data_path2', type=str, default="dataset/all_Hugoton_Panoma.xlsx", help='Hugoton_Panoma_path.')
    parser.add_argument('--data_path3', type=str, default="dataset/part_Daqing.xlsx", help='part_Daqing_path.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for Dataset split.')


    parser.add_argument('--num_classes1', type=int, default=5, help='Number of classes or categories in the daqing dataset')
    parser.add_argument('--num_classes2', type=int, default=9, help='Number of classes or categories in the  Hugoton_Panoma dataset')
    parser.add_argument('--num_classes3', type=int, default=5, help='Number of classes or categories in the  part_Daqing dataset')
    parser.add_argument('--features1', type=int, default=11, help='feature number for daqing.')
    parser.add_argument('--features2', type=int, default=7, help='feature number for Hugoton_Panoma.')
    parser.add_argument('--features3', type=int, default=13, help='feature number for Hugoton_Panoma.')
    parser.add_argument('--weights1', nargs='+', type=float, default=[0.4, 0.4, 0.43, 0.6, 0.5],
                        help='Focal loss weights for dataset 1')
    parser.add_argument('--weights2', nargs='+', type=float, default=[3.2, 1.0, 1.2, 3.5, 3.3, 1.5, 4.0, 1.1, 3.8],
                        help='Focal loss weights for dataset 2')
    parser.add_argument('--weights3', nargs='+', type=float, default=[0.5, 0.5, 0.4, 0.4, 0.5],
                        help='Focal loss weights for blind')
    parser.add_argument('--weights4', nargs='+', type=float, default=[2.93, 4.71, 3.28, 15.67, 19.85, 36.18],
                        help='Focal loss weights for blind')
    parser.add_argument('--hidden_size', type=int, default=128, help='BiLSTM hidden size.')
    parser.add_argument('--num_layers', type=int, default=1, help='BiLSTM and BiGRU num_layers.')
    parser.add_argument('--time_step_size', type=int, default=20, help='BiLSTM and BiGRU time step size.')
    parser.add_argument('--blind_well1', type=str, default="蛟11", help='Well name for blind1 well 1.')
    parser.add_argument('--blind_well2', type=str, default="板42", help='Well name for blind1 well 2.')
    parser.add_argument('--blind_well3', type=str, default="STUART", help='Well name for blind2 well')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

# 解析命令行参数
args = parse_arguments()


def print_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} | Shape: {param.shape} | Params: {param.numel():,}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total:,}")

def convert_to_tensor(data_list):
    """
    将列表中的数据转换为 PyTorch 浮点张量（Tensor）。

    参数：
    - data_list: 包含多个数组或列表的数据（list of list/array）。

    返回：
    - 转换后的 PyTorch 张量列表（list of torch.Tensor）。
    """
    return [torch.tensor(data, dtype=torch.float32) for data in data_list]



class MultiScaleDataset(Dataset):
    """
       多尺度数据集类，将不同尺度的特征数据组织成 PyTorch 数据集格式。

       作用：
       - 该类用于存储和组织多尺度特征数据，每个样本在不同尺度（例如 1、3、5）下的特征会被封装成一个元组，并与对应的标签配对，以便模型在训练时能够同时利用不同尺度的信息。

       数据格式：
       - self.data_by_scale: 一个列表，每个元素是不同尺度的特征数据，每个尺度的数据样本数量相同。
       - self.labels: 一个列表，存储所有样本的标签，每个样本的不同尺度数据共享同一个标签。

       参数：
       - data_scale_features: 以不同尺度组织的特征数据，列表格式，每个列表元素对应一个尺度的样本集。
       - labels: 样本对应的标签列表。

       方法：
       - __len__(): 返回数据集中样本的数量（即标签的数量）。
       - __getitem__(index): 返回索引 index 处的样本，包括 (尺度1样本, 尺度3样本, 尺度5样本) 及对应的标签。

       示例：
       假设 data_by_scale 结构如下：
       ```
       data_by_scale = [
           [样本0_尺度1, 样本1_尺度1, ..., 样本N_尺度1], # 尺度1
           [样本0_尺度3, 样本1_尺度3, ..., 样本N_尺度3], # 尺度3
           [样本0_尺度5, 样本1_尺度5, ..., 样本N_尺度5]  # 尺度5
       ]
       labels = [标签0, 标签1, ..., 标签N]
       ```
       当 index = 2 时，返回：
       ```
       (样本2_尺度1, 样本2_尺度3, 样本2_尺度5), 标签2
       ```
       这样模型可以同时接收多个尺度的数据进行训练。
       """
    def __init__(self, data_scale_features, labels):
        self.data_by_scale = data_scale_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return tuple(data[index] for data in self.data_by_scale), self.labels[index]


def prepare_multiscale_data(features):
    """
    生成不同尺度（1、3、5）的特征数据，并转换为 PyTorch 张量。

    参数：
    - features: 原始特征数据（NumPy 数组），形状为 (样本数, 特征数)。

    处理过程：
    1. 创建 new_data_scale_features，用于存储不同尺度的特征数据：
       - 尺度1：当前样本的特征。
       - 尺度3：当前样本 + 前后各一个样本（边界处用零填充）。
       - 尺度5：当前样本 + 前后各两个样本（边界处用零填充）。
    2. 遍历所有样本，为每个样本生成不同尺度的数据，并存入相应的列表中。
    3. 将列表转换为 PyTorch 张量，以便后续模型使用。

    返回：
    - new_data_by_scale: 包含不同尺度特征的 PyTorch 张量列表。
    """
    scales = [1, 3, 5]
    #创建一个列表new_data_by_scale，其中包含三个空列表，用于存储在不同尺度上的数据。这三个子列表对应于scales列表中的三个尺度。
    new_data_scale_features = [[] for _ in scales]
    for i in range(len(features)):
        #对于尺度1，直接将数据中的每个点添加到new_data_by_scale[0]中。
        new_data_scale_features[0].append(features[i])

        #对于尺度3，根据数据点的位置在new_data_by_scale[1]中构建包含当前点及其前一个和后一个点的数据。
        # 对于第一个数据点，没有前一个点，用零填充。对于最后一个数据点，没有后一个点，同样用零填充。
        if i == 0:
            new_data_scale_features[1].append(np.vstack([np.zeros(features.shape[1]), features[i], features[min(i + 1, len(features) - 1)]]))
        elif i == len(features) - 1:
            new_data_scale_features[1].append(np.vstack([features[max(i - 1, 0)], features[i], np.zeros(features.shape[1])]))
        else:
            new_data_scale_features[1].append(np.vstack([features[i - 1], features[i], features[i + 1]]))

        #对于尺度5，根据数据点的位置在new_data_by_scale[2]中构建包含当前点及其前两个和后两个点的数据。
        # 对于第一个数据点，没有前两个点，用零填充。对于最后一个数据点，没有后两个点，同样用零填充。
        left_padding_2 = np.zeros(features.shape[1]) if i - 2 < 0 else features[i - 2]
        left_padding_1 = np.zeros(features.shape[1]) if i - 1 < 0 else features[i - 1]
        right_padding_1 = np.zeros(features.shape[1]) if i + 1 >= len(features) else features[i + 1]
        right_padding_2 = np.zeros(features.shape[1]) if i + 2 >= len(features) else features[i + 2]
        new_data_scale_features[2].append(
            np.vstack([left_padding_2, left_padding_1, features[i], right_padding_1, right_padding_2]))

    #将数据从Numpy数组的形式转换为张量
    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale


def generate_multiscale_data(data, labels):
    """
    生成多尺度数据，并进行标准化和训练/测试集划分。

    作用：
    - 该方法首先对输入的原始数据进行多尺度转换，生成不同尺度的特征数据。
    - 对每个尺度的数据进行标准化，以保证不同尺度的数据分布一致。
    - 将数据按照 test_size 的比例划分为训练集和测试集，以供模型训练和评估。

    参数：
    - data: 原始特征数据（Tensor）。
    - labels: 样本对应的标签（列表）。

    返回：
    - train_test_data: 列表，包含以下四个部分：
      1. X_train: 训练集特征（包含不同尺度）。
      2. X_test: 测试集特征（包含不同尺度）。
      3. y_train: 训练集标签。
      4. y_test: 测试集标签。

    处理过程：
    1. 使用 `prepare_multiscale_data(data)` 生成不同尺度的特征数据，每个尺度的数据存储在 `new_data_by_scale` 中。
    2. 遍历 `new_data_by_scale`:
       - 先将数据重塑为二维数组，以便进行标准化处理。
       - 使用 `StandardScaler` 进行标准化，使数据均值为 0，方差为 1。
       - 重新调整数据形状，使其符合 (N, C, L) 形式：
         - N: 样本数
         - C: 通道数（不同尺度的通道数不同）
         - L: 每个样本的特征长度
    3. 使用 `train_test_split` 将每个尺度的数据随机划分为训练集（70%）和测试集（30%）。
    4. `X_train` 和 `X_test` 分别存储所有尺度的训练数据和测试数据。
    5. `y_train` 和 `y_test` 存储划分后的标签。
    6. 由于所有尺度的数据共享相同的标签，`y_train` 和 `y_test` 各包含三个相同的标签列表。

    示例：
    ```
    train_test_data = generate_multiscale_data(data, labels)
    X_train, X_test, y_train, y_test = train_test_data
    ```
    """
    new_data_by_scale = prepare_multiscale_data(data)
    j = 1
    X_train, X_test, y_train, y_test = [], [], [], []
    train_test_data = []
    # 将输入的多尺度数据进行预处理，并划分成训练集和测试集
    for i in range(len(new_data_by_scale)):
        # 把每个尺度的数据重塑为一个二维数组以便于标准化
        reshaped_data = new_data_by_scale[i] \
            .reshape(new_data_by_scale[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        # 将每个尺度的数据变换为维度为(N,C,L)的张量，其中N为样本数，C为通道数，L为特征长度
        if i == 0:
            new_data_by_scale[i] = new_data_by_scale[i].unsqueeze(
                1)  # 对于尺度1在第二个维度上增加一个大小为1的维度，因为unsqueeze()函数新增维度的大小固定为1
        else:
            new_data_by_scale[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale[i].size(0), j, new_data_by_scale[i].size(2))
        j = j + 2
        # 随机分割，原本数据集中的顺序会被打乱
        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(new_data_by_scale[i], labels,
                                                                                test_size=args.test_size,
                                                                                random_state=35)
        X_train.append(X_train_data)
        X_test.append(X_test_data)
        y_train.append(y_train_data)
        y_test.append(y_test_data)
    train_test_data.append(X_train)
    train_test_data.append(X_test)
    train_test_data.append(y_train)
    train_test_data.append(y_test)
    return train_test_data


def generate_multiscale_blind(features, labels):
    """
    生成多尺度数据（盲井预测场景），并进行标准化处理。

    作用：
    - 该方法对输入的原始特征数据进行多尺度转换，生成不同尺度的特征数据。
    - 对每个尺度的数据进行标准化，以保证数据分布一致，提升模型的泛化能力。
    - 返回处理后的多尺度数据和对应的标签，供模型进行训练或预测。

    参数：
    - features: 原始特征数据（Tensor）。
    - labels: 样本对应的标签（列表）。

    返回：
    - new_data_by_scale: 处理后的多尺度特征数据（列表，每个元素对应一个尺度）。
    - labels: 与输入数据对应的标签（未修改）。
    """
    new_data_by_scale = prepare_multiscale_data(features)
    j=1
    for i in range(len(new_data_by_scale)):
        reshaped_data = new_data_by_scale[i] \
            .reshape(new_data_by_scale[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        if i == 0:
            new_data_by_scale[i] = new_data_by_scale[i].unsqueeze(1)
        else:
            new_data_by_scale[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale[i].size(0), j, new_data_by_scale[i].size(2))
        j = j + 2
    return new_data_by_scale, labels


def get_confusion_matrix(trues, preds):
    """
    计算并归一化混淆矩阵。

    作用：
    - 该方法计算真实标签与预测标签之间的混淆矩阵，并对其按行进行归一化，以便分析分类模型的表现。
    - 归一化后的矩阵表示每个类别的预测分布，方便直观理解分类器的错误率和正确率。

    参数：
    - trues: 真实标签（列表或数组）。
    - preds: 预测标签（列表或数组）。

    返回：
    - 归一化后的混淆矩阵（百分比格式，数值范围 0-100，保留两位小数）。
    """
    conf_matrix = confusion_matrix(trues, preds)
    # 按行归一化为小数（0-1范围）
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以0
    conf_matrix_normalized = conf_matrix / row_sums
    return np.round(conf_matrix_normalized * 100, 2)  # 乘以100后保留两位小数



def write_file(file_path,predicted):
    """
    该函数用于将预测结果写入指定的文件中。每个预测结果以换行符分隔，逐行写入文件。

    参数：
    file_path (str)：目标文件的路径，预测结果将被写入到该文件中。

    predicted (iterable)：包含预测结果的可迭代对象（如列表或 PyTorch 张量）。每个元素表示一个预测结果，item() 方法将用于获取预测值。

    返回值：
    该方法没有返回值。它的作用是将预测结果逐行写入文件。
    """
    with open(file_path, "w") as file:
        for prediction in predicted:
            file.write(f"{prediction.item()}\n")
        file.close()

# 保存混淆矩阵
def save_matrix(conf_matrix_path,conf_matrix):
    """
    保存给定的混淆矩阵到指定的文件路径。

    参数:
    conf_matrix_path (str): 目标文件的路径，保存混淆矩阵的文件位置。
    confusion_matrix (array-like): 需要保存的混淆矩阵，通常是一个二维的数组或矩阵。

    返回值:
    无返回值，该方法将混淆矩阵保存为文件。
    """
    with open(conf_matrix_path, 'wb') as f:
        pickle.dump(conf_matrix, f)


def save_metrics_plot(accuracy, precision, recall, f1, run_num, save_dir, model_name):
    """
    可视化评估指标并保存为图片
    参数:
        accuracy (float): 准确率
        precision (float): 精确率
        recall (float): 召回率
        f1 (float): F1分数
        run_num (int): 运行次数
        save_dir (str): 保存路径
        model_name (str): 模型名称

    返回值:
        无返回值，该方法将绘制的评估指标保存为图片并展示。
    """
    # 创建绘图对象
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)

    # 隐藏坐标轴
    ax.axis('off')

    # 创建表格数据
    cell_text = [
        [f"{accuracy:.4f}"],
        [f"{precision:.4f}"],
        [f"{recall:.4f}"],
        [f"{f1:.4f}"]
    ]

    # 绘制表格
    table = ax.table(
        cellText=cell_text,
        rowLabels=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        colLabels=['Value'],
        loc='center',
        cellLoc='center'
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # 设置标题
    plt.title(f'Evaluation Metrics - Run {run_num} for {model_name} on {args.dataset}', fontsize=14, pad=20)

    # 自动调整布局并保存
    plt.tight_layout()
    save_path = save_dir + f'{model_name}_run{run_num}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()



class TimeSeriesDataset(Dataset):
    """
    用于处理时间序列数据的 PyTorch 数据集类，适用于基于 LSTM 等时序模型的训练。

    参数：
    - data: 形状为 (total_samples, input_size) 的时间序列数据，total_samples 表示样本总数，input_size 表示每个时间步的特征数。
    - labels: 形状为 (total_samples,) 或 (total_samples, target_dim) 的标签数据，其中每个样本对应一个标签。
    - seq_len: 序列长度，表示 LSTM 等模型需要的时间步数。

    属性：
    - data: 存储的时间序列数据。
    - labels: 存储的标签数据。
    - seq_len: 序列的长度，LSTM 输入的时间步数。

    方法：
    - __len__(): 返回数据集中的样本数量，计算公式为 `total_samples - seq_len + 1`，表示可以从数据中提取的时间序列数量。
    - __getitem__(self, idx): 获取索引 idx 对应的时间序列数据和标签：
        - 取出从索引 idx 开始的 seq_len 个时间步的数据。
        - 标签为最后一个时间步的标签（即标签是对应序列最后的值）。
        - 返回一个元组 `(x, y)`，其中 `x` 为时间序列数据，`y` 为标签。
    """
    def __init__(self, data, labels, seq_len):
        """
        data: 形状为 (total_samples, input_size)
        labels: 形状为 (total_samples,) 或 (total_samples, target_dim)
        seq_len: LSTM 需要的时间步长度
        """
        self.data = data
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        # 样本数量 = 总数据长度 - 序列长度 + 1
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        # 取出 idx 开始的 seq_len 个时间步
        x = self.data[idx: idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]  # 取最后一个时间步的标签
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)



def get_time_series_dataset(data, labels, seq_len):
    """
    获取时间序列数据的特征和标签数据

    data: 形状为 (total_samples, input_size)
    labels: 形状为 (total_samples,) 或 (total_samples, target_dim)
    seq_len: LSTM 需要的时间步长度

    返回：
    - X: 时间序列数据集特征数据（Tensor）
    - y: 时间序列数据集标签数据（Tensor）
    """

    #获取时间序列数据集
    dataset = TimeSeriesDataset(data, labels, seq_len)
    # 获取所有特征 X 和标签 y
    X = torch.stack([dataset[i][0] for i in range(len(dataset))])  # 假设 dataset[i] 返回 (feature, label)
    y = torch.tensor([dataset[i][1] for i in range(len(dataset))])  # 适用于单一标量标签
    return X, y

#定义一个数据分割方法，根据井名将每个井分别取20%作为测试集，其余作为训练集，并返回X_train, X_test, y_train, y_test


def split_by_well1(X, y, well_column: str, test_size: float = 0.2, random_state: int = 42, shuffle: bool = True):
    """
    根据井名分割数据集，每口井取 test_size 比例的数据作为测试集，其余作为训练集。

    参数：
    - X: 特征数据（DataFrame）
    - y: 岩性标签（Series）
    - well_column: 井名列名（str）
    - test_size: 测试集比例，默认 0.2
    - random_state: 随机种子，保证可复现性
    - shuffle: 是否打乱数据，默认 True

    返回：
    - X_train, X_test, y_train, y_test
    """
    train_idx, test_idx = [], []

    # 遍历每口井，分别划分训练集和测试集
    for well in X[well_column].unique():
        well_indices = X[X[well_column] == well].index
        if shuffle:
            train_idx_well, test_idx_well = train_test_split(
                well_indices, test_size=test_size, random_state=random_state
            )
        else:
            train_idx_well, test_idx_well = train_test_split(
            well_indices, test_size=test_size, shuffle=False
        )
        train_idx.extend(train_idx_well)
        test_idx.extend(test_idx_well)

    # 根据索引划分数据
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # 去除井名列，防止模型学习井名信息
    X_train = X_train.drop(columns=[well_column])
    X_test = X_test.drop(columns=[well_column])

    return X_train.values, X_test.values, y_train.values, y_test.values
