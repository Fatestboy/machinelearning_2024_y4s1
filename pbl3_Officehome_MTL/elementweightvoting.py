# 테스트 데이터셋에 대한 정확도 계산 함수
import torch
import csv
import torch.optim       as optim
import torch.nn as nn
import torch.cuda
from torch.nn.functional import normalize
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.scalarWeight_0 = nn.Parameter(torch.tensor(1.0))
        self.scalarWeight_1 = nn.Parameter(torch.tensor(1.0))
        self.scalarWeight_2 = nn.Parameter(torch.tensor(1.0))

        self.arrCWeight_0 = nn.Parameter(torch.tensor([[1.0]*65]))
        self.arrDWeight_0 = nn.Parameter(torch.tensor([[1.0]*4]))
        self.arrCWeight_1 = nn.Parameter(torch.tensor([[1.0]*65]))
        self.arrDWeight_1 = nn.Parameter(torch.tensor([[1.0]*4]))
        self.arrCWeight_2 = nn.Parameter(torch.tensor([[1.0]*65]))
        self.arrDWeight_2 = nn.Parameter(torch.tensor([[1.0]*4]))

        self.CLinear = nn.Sequential(
            nn.Linear(65, 65),
            #nn.BatchNorm1d(5),
            #nn.ReLU(),
            #nn.Linear(5, 65)
        )
        self.DLinear = nn.Sequential(
            nn.Linear(4, 4),
            #nn.BatchNorm1d(5),
            #nn.ReLU(),
            #nn.Linear(5, 4)
        )

    def forward(self, categoryProbs, domainProbs):
        #sumCategoryProbs = self.scalarWeight_0 * categoryProbs[0] + self.scalarWeight_1 * categoryProbs[1] + self.scalarWeight_2 * categoryProbs[2]
        #sumDomainProbs = self.scalarWeight_0 * domainProbs[0] + self.scalarWeight_1 * domainProbs[1] + self.scalarWeight_2 * domainProbs[2]
        sumCategoryProbs = self.arrCWeight_0 * categoryProbs[0] + self.arrCWeight_1 * categoryProbs[1] + self.arrCWeight_2 * categoryProbs[2]
        sumDomainProbs = self.arrDWeight_0 * domainProbs[0] + self.arrDWeight_1 * domainProbs[1] + self.arrDWeight_2 * domainProbs[2]
        #return self.CLinear(sumCategoryProbs), self.DLinear(sumDomainProbs)
        return sumCategoryProbs, sumDomainProbs

def get_dataset(dataDir='effnetb1_30_train_prob.csv'):
    probCategory_1 = list()
    probCategory_2 = list()
    probCategory_3 = list()
    probDomain_1 = list()
    probDomain_2 = list()
    probDomain_3 = list()
    labelCategory = list()
    labelDomain = list()

    with open(dataDir, 'r') as f:
        rdr = list(csv.reader(f))
        rdr.pop(0)

        didx_1 = 4
        cidx_1 = didx_1 + 65

        didx_2 = cidx_1 + 4
        cidx_2 = didx_2 + 65

        didx_3 = cidx_2 + 4
        cidx_3 = didx_3 + 65

        for row in rdr:
            row = [float(element) for element in row]
            probDomain_1.append(row[:didx_1])
            probCategory_1.append(row[didx_1:cidx_1])
            probDomain_2.append(row[cidx_1:didx_2])
            probCategory_2.append(row[didx_2:cidx_2])
            probDomain_3.append(row[cidx_2:didx_3])
            probCategory_3.append(row[didx_3:cidx_3])
            labelDomain.append(int(row[-2]))
            labelCategory.append(int(row[-1]))

    probCategory_1 = torch.tensor(probCategory_1)
    probCategory_2 = torch.tensor(probCategory_2)
    probCategory_3 = torch.tensor(probCategory_3)
    probDomain_1   = torch.tensor(probDomain_1)
    probDomain_2   = torch.tensor(probDomain_2)
    probDomain_3   = torch.tensor(probDomain_3)
    labelCategory  = torch.tensor(labelCategory)
    labelDomain    = torch.tensor(labelDomain)
    return probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain


def calculate_accuracy(probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain):
    # 각 클래스(카테고리와 도메인)에 대한 예측 결과를 저장할 리스트 초기화
    predicted_categories = []
    predicted_domains = []

    # 모델을 사용하여 각 테스트 샘플에 대한 예측값 생성
    for i in range(len(labelCategory)):
        # 각 클래스에 대한 확률값에서 가장 높은 확률을 가진 인덱스를 예측값으로 사용
        pred_cat = torch.argmax(probCategory_1[i] + probCategory_2[i] + probCategory_3[i])
        pred_dom = torch.argmax(probDomain_1[i] + probDomain_2[i] + probDomain_3[i])
        predicted_categories.append(pred_cat)
        predicted_domains.append(pred_dom)

    # 예측 결과와 실제 레이블을 비교하여 정확도 계산
    correct_category = torch.sum(torch.tensor(predicted_categories) == labelCategory).item()
    correct_domain = torch.sum(torch.tensor(predicted_domains) == labelDomain).item()
    total_samples = len(labelCategory)

    accuracy_category = (correct_category / total_samples)*100
    accuracy_domain = (correct_domain / total_samples)*100
    print(accuracy_category, accuracy_domain)


def train_model(model, probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain, num_epochs=100, lr=0.1):
    # 손실 함수 정의: 카테고리와 도메인의 손실을 합산
    def custom_loss(outputs, labelCategoryIdx, labelDomainIdx):
        labelCategory = list()
        labelDomain = list()

        for curIter in labelCategoryIdx:
            temp = [0]*65
            temp[curIter] = 1.0
            labelCategory.append(temp)
        for curIter in labelDomainIdx:
            temp = [0]*4
            temp[curIter] = 1.0
            labelDomain.append(temp)

        labelCategory = torch.tensor(labelCategory).cuda()
        labelDomain = torch.tensor(labelDomain).cuda()

        loss_category = torch.nn.functional.cross_entropy(outputs[0], labelCategory)
        loss_domain = torch.nn.functional.cross_entropy(outputs[1], labelDomain)
        return loss_category + loss_domain

    # 정확도 계산 함수 정의
    def calculate_accuracy(outputs, labelCategoryIdx, labelDomainIdx):
        # 카테고리와 도메인의 예측값 계산
        predicted_category = torch.argmax(outputs[0], dim=1)
        predicted_domain = torch.argmax(outputs[1], dim=1)

        # 실제 라벨과 예측값 비교하여 정확도 계산
        correct_category = torch.sum(predicted_category == labelCategoryIdx).item()
        correct_domain = torch.sum(predicted_domain == labelDomainIdx).item()
        total_samples = len(labelCategoryIdx)

        accuracy_category = (correct_category / total_samples) * 100
        accuracy_domain = (correct_domain / total_samples) * 100

        return accuracy_category, accuracy_domain

    # 손실 함수 및 최적화 기준 설정
    criterion = custom_loss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 텐서를 CUDA로 이동
    model.cuda()
    probCategory_1 = probCategory_1.cuda()
    probCategory_2 = probCategory_2.cuda()
    probCategory_3 = probCategory_3.cuda()
    probDomain_1 = probDomain_1.cuda()
    probDomain_2 = probDomain_2.cuda()
    probDomain_3 = probDomain_3.cuda()
    labelCategory = labelCategory.cuda()
    labelDomain = labelDomain.cuda()

    # 훈련
    dataset_size = len(probCategory_1)
    batch_size = 32
    for epoch in range(num_epochs):
        # 각 에폭마다 데이터셋을 섞음
        indices = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]

            # 현재 미니배치 선택
            batch_probCategory_1 = probCategory_1[batch_indices]
            batch_probCategory_2 = probCategory_2[batch_indices]
            batch_probCategory_3 = probCategory_3[batch_indices]
            batch_probDomain_1 = probDomain_1[batch_indices]
            batch_probDomain_2 = probDomain_2[batch_indices]
            batch_probDomain_3 = probDomain_3[batch_indices]
            batch_labelCategory = labelCategory[batch_indices]
            batch_labelDomain = labelDomain[batch_indices]

            # 모델 예측
            outputs = model([batch_probCategory_1, batch_probCategory_2, batch_probCategory_3],
                            [batch_probDomain_1, batch_probDomain_2, batch_probDomain_3])

            # 손실 계산
            loss = criterion(outputs, batch_labelCategory, batch_labelDomain)

            # 기울기 초기화 및 역전파
            optimizer.zero_grad()
            loss.backward()

            # 가중치 업데이트
            optimizer.step()

        # 전체 데이터셋에 대한 정확도 계산
        outputs = model([probCategory_1, probCategory_2, probCategory_3], [probDomain_1, probDomain_2, probDomain_3])
        accuracy_category, accuracy_domain = calculate_accuracy(outputs, labelCategory, labelDomain)

        # 로그 출력
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Category Accuracy: {accuracy_category}%, Domain Accuracy: {accuracy_domain}%')

    # 훈련된 가중치와 최종 정확도 반환
    return model.scalarWeight_0.item(), model.scalarWeight_1.item(), model.scalarWeight_2.item(), accuracy_category, accuracy_domain


    # 손실 함수 및 최적화 기준 설정
    criterion = custom_loss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 훈련
    dataset_size = len(probCategory_1)
    batch_size = 32
    for epoch in range(num_epochs):
        # 각 에폭마다 데이터셋을 섞음
        indices = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]

            # 현재 미니배치 선택
            batch_probCategory_1 = probCategory_1[batch_indices]
            batch_probCategory_2 = probCategory_2[batch_indices]
            batch_probCategory_3 = probCategory_3[batch_indices]
            batch_probDomain_1 = probDomain_1[batch_indices]
            batch_probDomain_2 = probDomain_2[batch_indices]
            batch_probDomain_3 = probDomain_3[batch_indices]
            batch_labelCategory = labelCategory[batch_indices]
            batch_labelDomain = labelDomain[batch_indices]

            # 모델 예측
            outputs = model([batch_probCategory_1, batch_probCategory_2, batch_probCategory_3],
                            [batch_probDomain_1, batch_probDomain_2, batch_probDomain_3])

            # 손실 계산
            loss = criterion(outputs, batch_labelCategory, batch_labelDomain)

            # 기울기 초기화 및 역전파
            optimizer.zero_grad()
            loss.backward()

            # 가중치 업데이트
            optimizer.step()

        # 전체 데이터셋에 대한 정확도 계산
        outputs = model([probCategory_1, probCategory_2, probCategory_3], [probDomain_1, probDomain_2, probDomain_3])
        accuracy_category, accuracy_domain = calculate_accuracy(outputs, labelCategory, labelDomain)

        # 로그 출력
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Category Accuracy: {accuracy_category}%, Domain Accuracy: {accuracy_domain}%')

    # 훈련된 가중치와 최종 정확도 반환
    return model.scalarWeight_0.item(), model.scalarWeight_1.item(), model.scalarWeight_2.item(), accuracy_category, accuracy_domain

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

def test_model(model, probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain):
    # 모델을 평가 모드로 전환
    model.eval()

    # 텐서를 CUDA로 이동
    probCategory_1 = probCategory_1.cuda()
    probCategory_2 = probCategory_2.cuda()
    probCategory_3 = probCategory_3.cuda()
    probDomain_1 = probDomain_1.cuda()
    probDomain_2 = probDomain_2.cuda()
    probDomain_3 = probDomain_3.cuda()
    labelCategory = labelCategory.cuda()
    labelDomain = labelDomain.cuda()

    # 모델 예측
    with torch.no_grad():
        outputs = model([probCategory_1, probCategory_2, probCategory_3], [probDomain_1, probDomain_2, probDomain_3])

    # 손실 계산 함수 정의 (cross-entropy)
    def custom_loss(outputs, labelCategory, labelDomain):
        loss_category = F.cross_entropy(outputs[0], labelCategory)
        loss_domain = F.cross_entropy(outputs[1], labelDomain)
        return loss_category + loss_domain

    # 정확도 계산 함수 정의
    def calculate_accuracy(outputs, labelCategory, labelDomain):
        # 카테고리와 도메인의 예측값 계산
        predicted_category = torch.argmax(outputs[0], dim=1)
        predicted_domain = torch.argmax(outputs[1], dim=1)

        # 실제 라벨과 예측값 비교하여 정확도 계산
        correct_category = torch.sum(predicted_category == labelCategory).item()
        correct_domain = torch.sum(predicted_domain == labelDomain).item()
        total_samples = len(labelCategory)

        accuracy_category = (correct_category / total_samples) * 100
        accuracy_domain = (correct_domain / total_samples) * 100

        return accuracy_category, accuracy_domain

    # mAP 계산 함수 정의
    def calculate_mAP(outputs, labelCategory, labelDomain):
        # 카테고리와 도메인의 예측값 계산
        pred_category = outputs[0].detach().cpu()
        pred_domain = outputs[1].detach().cpu()
        label_category = labelCategory.detach().cpu()
        label_domain = labelDomain.detach().cpu()

        # Calculate mAP for categories
        mAP_category = average_precision_score(F.one_hot(label_category, num_classes=65).numpy(), pred_category.numpy(), average='macro')

        # Calculate mAP for domains
        mAP_domain = average_precision_score(F.one_hot(label_domain, num_classes=4).numpy(), pred_domain.numpy(), average='macro')

        return mAP_category, mAP_domain

    # 손실 계산
    loss = custom_loss(outputs, labelCategory, labelDomain)

    # 정확도 계산
    accuracy_category, accuracy_domain = calculate_accuracy(outputs, labelCategory, labelDomain)

    # mAP 계산
    mAP_category, mAP_domain = calculate_mAP(outputs, labelCategory, labelDomain)

    # 결과 출력
    print(f'Test Loss: {loss.item()}')
    print(f'Category Accuracy: {accuracy_category}%, Domain Accuracy: {accuracy_domain}%')
    print(f'Category mAP: {mAP_category}, Domain mAP: {mAP_domain}')

    return loss.item(), accuracy_category, accuracy_domain, mAP_category, mAP_domain

if __name__ == "__main__":
    model = Model()
    # 데이터셋 불러오기
    probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain = get_dataset(dataDir='effnetb2_30_test_prob.csv')
    calculate_accuracy(probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain)
    trained_weights = train_model(model, probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain)

    probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain = get_dataset('effnetb2_30_test_prob.csv')
    calculate_accuracy(probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain)
    trained_weights = test_model(model, probCategory_1, probCategory_2, probCategory_3, probDomain_1, probDomain_2, probDomain_3, labelCategory, labelDomain)
    # 훈련된 가중치 확인
    print("Trained Weights:", trained_weights)
