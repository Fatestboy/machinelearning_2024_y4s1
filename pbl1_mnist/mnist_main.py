import argparse
import torch
import torch.nn            as nn
import torch.optim         as optim
import os
from torchvision          import datasets, transforms
import time


#cmd args define
parser = argparse.ArgumentParser(description='MLP-MNIST')
parser.add_argument('--batch-size', type=int, default=100, metavar='batch_size',
                    help='batch-size (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='test',
                    help='validation batch size (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='epochs',
                    help='epochs (default: 100)')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='lr',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--save', type=str, default='mnist_model', metavar='save',
                    help='save file name')
parser.add_argument('--node-ratio', type=int, default=1, metavar='node_ratio',
                    help='hidden layer fain_in : 128*node (default: 1)')
parser.add_argument('--hidden-layers', type=int, default=5, metavar='hidden_layers',
                    help='nuber of hidden layers (defailt: 5)')
parser.add_argument('--train-or-eval', type=str, default='train', metavar='evalation or training',
                    help='is code act in evaluation or training (train || eval)')
###################################################################################################
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# MNIST data load and split data sets for taining, validation and evaluation
def init_dataset():
    global train_loader
    global val_loader
    global eval_loader
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.15,), (0.3,))
                                    ]))

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size 

    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    # training datasets, validation datasets

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)

    eval_loader = torch.utils.data.DataLoader(
                       datasets.MNIST('./data', train=False, 
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.15,), (0.3,))
                   ])), batch_size=args.test_batch_size, shuffle=False)
    # evaluation datasets
    
    


class MLP(nn.Module):
    def __init__(self, layers, node_ratio, level):
        self.layers = layers
        self.node_ratio = node_ratio
        self.level = level
    def make_layer(self):
        '''
        This method returns multi hidden layers by "args.hidden_layers"
        Also note that node size is defined as 128 * args.node_ratio. For example, if you set node_ratio as 3, hidden layer's node
        For example, if you set args.hidden_layers as 3 and args.node_ratio as 2, MLP model is implemented as (28^2x256)x(256x256)x(256x256)x(256x256)x(256x10)
        '''
        
        layers=[]
        for _ in range(self.layers):
            layers.append(nn.Sequential(
                nn.Linear(64*self.node_ratio, 64*self.node_ratio, self.level),
                nn.BatchNorm1d(64*self.node_ratio),
                nn.Hardtanh()
            ))
        return nn.Sequential(*layers)
        
        
class MLP_MNIST(MLP):
    def __init__(self, layers, node_ratio, level):
        super(MLP, self).__init__()
        self.node_ratio = node_ratio
        self.layers = layers
        self.level = level
        
        self.fc_in = nn.Linear(28*28, 64*self.node_ratio, self.level)
        self.bn = nn.BatchNorm1d(64*self.node_ratio)
        self.Relu = nn.ReLU()
        
        self.hiddenLayers = self.make_layer()

        self.fc_out = nn.Linear(64*self.node_ratio, 10, self.level)

        self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        
        x = self.fc_in(x)
        x = self.bn(x)
        x = self.Relu(x)
        
        x = self.hiddenLayers(x)
        
        x = self.fc_out(x)

        return self.logsoftmax(x)

def train(epoch, model):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def validation (epoch, model):
    global best_acc
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_acc = 100.*correct/total
    acc = 100.*correct/total
    
    state = {
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('mnist_result'):
        os.mkdir('mnist_result')
    if acc > best_acc:
        torch.save(state, os.path.join('./mnist_result/', args.save))
        best_acc = acc

    return val_loss, val_acc


def evaluation(model, eval_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in eval_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc

criterion = nn.CrossEntropyLoss()

def training(model, optimizer):
    start_e_time = time.time()

    for epoch in range(0, args.epochs):
        start_p_time = time.time()
        print('cur train level: ',model.level,' cur train epoch:', epoch)
        train_loss, train_acc = train(epoch, model)
        test_loss,  val_acc  = validation(epoch,  model)
        time.sleep(0)
        print('Epoch: %d/%d | LR: %.4f | Train Loss: %.3f | Train Acc: %.2f | Test Loss: %.3f | Test Acc: %.2f (%.2f)' %
        (epoch+1, args.epochs, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, val_acc, best_acc))
        print('Spent time per epoch : {:.4f}sec'.format((time.time() - start_p_time)))

    time.sleep(0)
    print('---------------------------- TRAINING DONE ------------------------------')
    print('--------------------------- best_acc = %.2f ----------------------------' % best_acc)
    print('---------------------------- Time : {:.1f}sec -----------------------------'.format((time.time() - start_e_time)))

def test():
    best_model_path = os.path.join('./mnist_result/', args.save)
    best_model_state = torch.load(best_model_path)
    best_model = MLP_MNIST(node_ratio=args.node_ratio, layers=args.hidden_layers, level=2)
    best_model.load_state_dict(best_model_state['model'])

    # Evaluate the best model
    eval_acc = evaluation(best_model, eval_loader)
    print('--------------------------- Best Model Test Acc: %.2f -----------------------------' % eval_acc)
    

##############################################EntryPoint##########################################
if __name__ == '__main__':
    init_dataset()
    model = MLP_MNIST(node_ratio=args.node_ratio, layers=args.hidden_layers, level = 2)
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    if args.cuda:
        torch.cuda.set_device(0)
        model.cuda()
    if args.train_or_eval == 'train': 
       training(model, optimizer) # if you set args.train_or_eval as train, branch to this code
    elif args.train_or_eval == 'eval':
        test() # if you set args.train_or_eval as eval, branch to this code
    
 
