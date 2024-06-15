import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 데이터셋을 위한 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet에 맞는 이미지 크기 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# CIFAR-100 훈련 및 테스트 데이터셋 로드
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


# EfficientNet 모델 로드 (예: 'efficientnet_b0') 및 디바이스로 이동

def soft_voting(eachOutputs):
    ensembleOut = torch.zeros_like(eachOutputs[0])
    for predIter in eachOutputs:
        ensembleOut += predIter
    ensembleOut /= len(eachOutputs)
    return ensembleOut


def weighted_voting(eachOutputs, weights):
    ensembleOut = torch.zeros_like(eachOutputs[0])
    for predIter, weightIter in zip(eachOutputs, weights):
        ensembleOut += weightIter * predIter
    ensembleOut /= len(eachOutputs)
    return ensembleOut


import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 데이터셋을 위한 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet에 맞는 이미지 크기 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# CIFAR-100 훈련 및 테스트 데이터셋 로드
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


# EfficientNet 모델 로드 (예: 'efficientnet_b0') 및 디바이스로 이동

def soft_voting(eachOutputs):
    ensembleOut = torch.zeros_like(eachOutputs[0])
    for outIter in eachOutputs:
        ensembleOut += outIter
    ensembleOut /= len(eachOutputs)
    return ensembleOut


def weighted_voting(eachOutputs, weights):
    ensembleOut = torch.zeros_like(eachOutputs[0])
    for outIter, weightIter in zip(eachOutputs, weights):
        ensembleOut += weightIter * outIter
    ensembleOut /= len(eachOutputs)
    return ensembleOut


def hard_voting(eachOutputs):
    def find_most_freq_of(input):
        cnts = [0] * 100
        for iter in input:
            cnts[iter] += 1
        return cnts.index(max(cnts))

    ensembelPred = []
    for idx in range(len(eachOutputs[0])):
        valuesAtIdx = [output[idx] for output in eachOutputs]
        maxCnts = find_most_freq_of(valuesAtIdx)
        ensembelPred.append(maxCnts)

    return torch.tensor(ensembelPred)


# 모델을 평가하는 함수
def evaluate(models, mode, ensembelWeights):
    eachOutputs = list()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, label in test_loader:
            images, label = images.to(device), label.to(device)
            for curModel in models:
                curModel.eval()
                eachOutputs.append(curModel(images))
            if mode == 'soft voting':
                outputSum = soft_voting(eachOutputs)
                _, predicted = torch.max(outputSum, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            elif mode == 'weighted voting':
                outputSum = weighted_voting(eachOutputs, ensembelWeights)
                _, predicted = torch.max(outputSum, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            elif mode == 'hard voting':
                predicted = hard_voting(eachOutputs)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            else:
                print('models you selected is not supported now')
                exit(1)
    print(f'Accuracy at {mode} : {100 * correct / total}%')


def main():
    #################################################################
    voteMode = 'soft voting'  # soft voting, weight voting, hard voting
    modelName = ['efficientnet_b2'
                 ]

    modelDir = ['./CIFAR_effnetb2_40_result.pth']
    ensembleWeights = [1, 9, 1, 0.5, 1, 1]
    #################################################################
    models = list()
    for modelNameIter, modelDirIter in zip(modelName, modelDir):
        curModel = timm.create_model(modelNameIter, pretrained=False, num_classes=100).to(device)
        # curModel.load_state_dict(torch.load(modelDirIter))
        curModel.to(device)
        models.append(curModel)

    evaluate(models=models, mode=voteMode,
             ensembelWeights=ensembleWeights)  # voting mode는 'soft voting','hard voting','weighted voting' 중 하나 사용 가능


if __name__ == '__main__':
    main()