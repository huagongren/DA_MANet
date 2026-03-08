""""
该模块定义了训练模型的函数，包括训练模型、评估模型、保存预测结果等功能。
除了LMAFNet模型的训练和评估，其他模型的训练和评估代码也在该模块中。
"""
import torch
def train_model(model, criterion, optimizer, data_loader, device):
    model.train()
    #print(f"Number of batches in train_loader: {len(data_loader)}")
    for input_data, target in data_loader:

        #print("input_data", input_data.shape)
        #print("target", target)
        #print("target", target.shape)

        input_data, target = input_data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input_data)
        _, predicted = torch.max(output, 1)
        #print("predicted", predicted)
        #print("output", output)
        #print("output", output.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            #print("predicted", predicted)
            all_predicted.extend(predicted.cpu().tolist())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    all_predicted = torch.tensor(all_predicted)
    return accuracy, all_predicted

def save_predictions(predictions, file_path):
    with open(file_path, "w") as file:
        for prediction in predictions:
            file.write(f"{prediction.item()}\n")
