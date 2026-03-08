"""
此模块是用于训练和评估LMAFNet模型的主函数

"""
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from util import *
from load_data import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# 导入自定义的多尺度网络模型和多目标焦点损失函数
from LMAFNet_model import MultiScaleNetwork, multi_FocalLoss
# 导入训练和评估模型的函数
from LMAFNet_train import train_model, evaluate


def main():
    # 解析命令行参数
    args = parse_arguments()

    # 根据不同的数据集加载训练和测试数据
    if args.dataset == "daqing":
        x_train, y_train, data_train_loader, x_test, y_test, data_test_loader = get_daqing_multiscale(args.data_path1)
        num_classes = args.num_classes1
        model_params = (num_classes, args.features1)
        focal_loss_weights = args.weights1
        save_path = 'datasave/daqing/'
    elif args.dataset == "Hugoton_Panoma":
        x_train, y_train, data_train_loader, x_test, y_test, data_test_loader = get_Hugoton_Panoma_multiscale(args.data_path2)
        num_classes = args.num_classes2
        model_params = (num_classes, args.features2)
        focal_loss_weights = args.weights2
        save_path = 'datasave/Hugoton_Panoma/'
    elif args.dataset == "part_daqing":
        x_train, y_train, data_train_loader, x_test, y_test, data_test_loader = get_part_daqing_multiscale(args.data_path3)
        num_classes = args.num_classes3
        model_params = (num_classes, args.features3)
        focal_loss_weights = args.weights4
        save_path = 'datasave/part_Daqing/'
    elif args.dataset == "blind1":
        x_train, y_train, data_train_loader, x_test, y_test, data_test_loader = (
            get_blind1_multiscale(args.data_path1, args.blind_well1, args.blind_well2))
        num_classes = args.num_classes1
        model_params = (num_classes, args.features1,0.6)
        focal_loss_weights = args.weights3
        save_path = 'datasave/blind1/'
    elif args.dataset == "blind2":
        x_train, y_train, data_train_loader, x_test, y_test, data_test_loader = (
            get_blind2_multiscale(args.data_path2, args.blind_well3))
        num_classes = args.num_classes2
        model_params = (num_classes, args.features2,0.6)
        focal_loss_weights = args.weights2
        save_path = 'datasave/blind2/'

    else:
        raise ValueError("Invalid dataset name")

    # 设置训练次数
    times = 1
    # 设置随机种子以确保结果的可重复性
    seed = args.seed
    # 初始化训练和测试准确率列表
    train_accs = []
    test_accs = []

    # 进行多次训练
    for j in range(times):
        # 设置随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # 初始化模型
        model = MultiScaleNetwork(*model_params)
        # 初始化优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # 初始化学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        # 初始化损失函数
        criterion = multi_FocalLoss(num_classes, focal_loss_weights)
        # 初始化最佳准确率和模型保存路径
        best_accuracy = 0
        best_model_path = save_path + f'{args.dataset}_best.pth'
        # 设置设备（GPU或CPU）
        device = torch.device("cuda:0" if args.cuda else "cpu")
        model.to(device)

        # 训练模型
        for epoch in range(1, args.epochs + 1):
            train_model(model, criterion, optimizer, data_train_loader, device=device)
            # 计算训练准确率
            train_accuracy, _ , _= evaluate(model, x_train, y_train, device=device, return_probabilities=False)
            # 计算测试准确率
            test_accuracy, _, predicted_test = evaluate(model, x_test, y_test, device=device, return_probabilities=False)
            train_accs.append(train_accuracy)
            test_accs.append(test_accuracy)

            # 如果当前测试准确率高于最佳准确率，保存模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), best_model_path)

            # 更新学习率调度器
            scheduler.step()

        # 加载最佳模型进行最终评估
        model.load_state_dict(torch.load(best_model_path))
        accuracy, _, predicted , probabilities= evaluate(model, x_test, y_test, device=device, return_probabilities=True)
        # 保存预测结果到文件
        prob_dir = os.path.dirname(save_path + f'y_prob/LMAFNet_run{j + 1}_prob.npy')
        true_dir = os.path.dirname(save_path + f'y_true/LMAFNet_run_true_labels_run{j + 1}.npy')
        pred_dir = os.path.dirname(save_path + f'y_pre/LMAFNet_run{j + 1}.txt')
        os.makedirs(prob_dir, exist_ok=True)
        os.makedirs(true_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        # 保存预测结果到文件
        path = save_path + f'y_pre/LMAFNet_run{j + 1}.txt'
        write_file(path, predicted)

        # 保存预测概率到文件（用于绘制ROC曲线）
        prob_path = save_path + f'y_prob/LMAFNet_run{j + 1}_prob.npy'
        # 将张量转换为numpy数组并保存
        np.save(prob_path, probabilities.cpu().numpy())

        # 如果需要，也可以保存真实标签
        true_path = save_path + f'y_true/LMAFNet_run_true_labels_run{j + 1}.npy'
        np.save(true_path, y_test.cpu().numpy())

        # 计算精确率、召回率和F1分数
        precision = precision_score(y_test.cpu(), predicted.cpu(), average='macro')
        recall = recall_score(y_test.cpu(), predicted.cpu(), average='macro')
        f1 = f1_score(y_test.cpu(), predicted.cpu(), average='macro')
        conf_matrix = get_confusion_matrix(y_test.cpu(), predicted.cpu())

        #可视化评估结果
        eval_save_path = save_path + f'model_evaluation/'
        save_metrics_plot(accuracy, precision, recall, f1, j + 1, eval_save_path, "LMAFNet")

        # 可视化混淆矩阵
        plt.figure(figsize=(10, 7))

        # 假设你有一个岩性名称列表，例如：
        lithology_labels = ['SM', 'DM', 'HS', 'Hgs', 'BS']  # 替换为你的实际岩性名称
        #lithology_labels = ["SS","CSiS","FSiS","SiSh","MS","WS","D","PS","BS"]  # 假设你只有5个岩性，则可以用数字代替

        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='.2f',  # 显示两位小数
            cmap='YlOrBr',
            xticklabels=lithology_labels,
            yticklabels=lithology_labels
        )
        plt.title(f'Confusion Matrix for LMAFNet on {args.dataset}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        confusion_matrix_save_path = save_path + f'confusion_matrix/'
        plt.savefig(confusion_matrix_save_path + f'LMAFNet_run_{j + 1}.png')
        plt.show()

        # 可视化训练和测试准确率
        plt.figure(figsize=(10, 7))
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(test_accs, label='Test Accuracy')
        plt.title(f'Accuracy Curves for LMAFNet on {args.dataset}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        accuracy_curve_save_path = save_path + f'accuracy_curve/'
        plt.savefig(accuracy_curve_save_path + f'LMAFNet_run_{j + 1}.png')
        plt.show()

# 如果当前脚本是主程序，运行main函数
if __name__ == '__main__':
    main()
