"""
该模块定义了训练和评估LMAFNet模型的函数，包括训练模型、评估模型、保存预测结果等功能
"""

import torch
def train_model(model, criterion, optimizer, data_loader, device):
    model.train()
    for input_data, target in data_loader:
        # print("input_data", input_data)
        # print("target", target)
        #因为在处理多尺度数据时会分开处理各个尺度的数据，为了将各个尺度的数据分开处理，这里需要将输入数据的格式转为List
        input_data, target = [x.to(device) for x in input_data], target.to(device)

        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, x_data, y_data, device, return_probabilities=True):
    """
    评估模型性能，返回整体准确率、各分类别准确率和预测结果
    Args:
        model: 待评估的模型
        x_data: 输入特征（多尺度数据）
        y_data: 真实标签
        device: 计算设备
        return_probabilities: 是否返回预测概率（用于绘制ROC曲线）

    Returns:
        overall_acc: 整体准确率
        class_acc_dict: 类别准确率字典 {class_id: accuracy}
        predicted: 预测类别
        probabilities: 预测概率（仅当return_probabilities=True时返回）
    """
    model.eval()
    with torch.no_grad():
        if device:
            x_data, y_data = [x.to(device) for x in x_data], y_data.to(device)

        # 模型预测
        output = model(x_data)

        # 如果需要返回概率，计算softmax概率
        if return_probabilities:
            probabilities = torch.softmax(output, dim=1)
        else:
            probabilities = None

        # 获取预测类别
        _, predicted = torch.max(output, 1)

        # 计算整体准确率
        correct = (predicted == y_data).sum().item()
        total = y_data.size(0)
        overall_acc = correct / total

        # 计算各分类别准确率
        class_acc_dict = {}
        unique_classes = torch.unique(y_data)

        for cls in unique_classes:
            # 创建类别掩码
            cls_mask = (y_data == cls)
            cls_total = cls_mask.sum().item()

            if cls_total == 0:  # 避免零除
                class_acc_dict[cls.item()] = 0.0
                continue

            # 统计当前类别的正确预测数
            cls_correct = ((predicted == cls) & cls_mask).sum().item()
            class_acc_dict[cls.item()] = cls_correct / cls_total

    if return_probabilities:
        return overall_acc, class_acc_dict, predicted, probabilities
    else:
        return overall_acc, class_acc_dict, predicted

def save_predictions(predictions, file_path):
    with open(file_path, "w") as file:
        for prediction in predictions:
            file.write(f"{prediction.item()}\n")
