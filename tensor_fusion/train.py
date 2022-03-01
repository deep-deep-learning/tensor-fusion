from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from .dataset import get_cmu_mosi_dataset
from .net import *
from .util import get_log_prior_coeff

def binary_map_train_TFN(learning_rate, epochs=50, batch_size=32, 
                         device=None, dtype=torch.float32, print_result=False):

    train_set, valid_set, test_set = get_cmu_mosi_dataset(binary=True, device=device, dtype=dtype)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=len(valid_set))
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))

    input_sizes = (300, 5, 20)
    hidden_sizes = (128, 32, 32)
    fusion_size = 128
    out_size = 1
    dropouts = (0.15, 0.15, 0.15, 0.15, 0.15)

    model = TFN(input_sizes, hidden_sizes, fusion_size, out_size, dropouts, device, dtype)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(list(model.parameters()), learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=print_result)
    
    max_accuracy = 0

    result = dict()
    result['train_loss'] = [float('inf')]
    result['valid_loss'] = [float('inf')]
    result['test_accuracy'] = [0.0]
    result['test_f1'] = [0.0]

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        
        for text, audio, vision, label in train_dataloader:
            model.zero_grad()
            output = model(text, audio, vision)
            loss = criterion(output, label)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss = train_loss / len(train_set)
        result['train_loss'].append(train_loss)
        
        model.eval()
        for text, audio, vision, label in valid_dataloader:
            output = model(text, audio, vision)
            valid_loss = criterion(output, label).item()
        valid_loss = valid_loss / len(valid_set)
        result['valid_loss'].append(valid_loss)
        scheduler.step(valid_loss)

        for text, audio, vision, label in test_dataloader:
            output = model(text, audio, vision)
        
        output = (output > 0).type(dtype)

        test_binary_accuracy = accuracy_score(label.cpu(), output.cpu())
        test_f1_score = f1_score(label.cpu(), output.cpu())
        result['test_accuracy'].append(test_binary_accuracy)  
        result['test_f1'].append(test_f1_score)
        
        if result['valid_loss'][-1] < min(result['valid_loss'][:-1]):
            max_accuracy = test_binary_accuracy
            model_state_dict = model.state_dict()

        if print_result:
            print('Epoch {}'.format(epoch))
            print('Train Loss {:.4f}'.format(train_loss))
            print('Valid Loss {:.4f}'.format(valid_loss))
            print('Test Bin Acc {:.4f}'.format(test_binary_accuracy))

    return max_accuracy, result, model_state_dict    


def binary_map_train_LMF(rank, learning_rate, epochs=50, batch_size=32, 
                         device=None, dtype=torch.float32, 
                         print_result=False):

    train_set, valid_set, test_set = get_cmu_mosi_dataset(binary=True, device=device, dtype=dtype)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=len(valid_set))
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))

    input_sizes = (300, 5, 20)
    hidden_sizes = (128, 32, 32)
    fusion_size = 128
    out_size = 1
    dropouts = (0.15, 0.15, 0.15, 0.15, 0.15)
    model = LMF(input_sizes, hidden_sizes, fusion_size, rank, out_size, dropouts, device, dtype)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(list(model.parameters()), learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=print_result)
    
    max_accuracy = 0

    result = dict()
    result['train_loss'] = [float('inf')]
    result['valid_loss'] = [float('inf')]
    result['test_accuracy'] = [0.0]
    result['test_f1'] = [0.0]

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        
        for text, audio, vision, label in train_dataloader:
            model.zero_grad()
            output = model(text, audio, vision)
            loss = criterion(output, label)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss = train_loss / len(train_set)
        result['train_loss'].append(train_loss)
        
        model.eval()
        for text, audio, vision, label in valid_dataloader:
            output = model(text, audio, vision)
            valid_loss = criterion(output, label).item()
        valid_loss = valid_loss / len(valid_set)
        result['valid_loss'].append(valid_loss)
        scheduler.step(valid_loss)

        for text, audio, vision, label in test_dataloader:
            output = model(text, audio, vision)
        
        output = (output > 0).type(dtype)

        test_binary_accuracy = accuracy_score(label.cpu(), output.cpu())
        test_f1_score = f1_score(label.cpu(), output.cpu())
        result['test_accuracy'].append(test_binary_accuracy)  
        result['test_f1'].append(test_f1_score)

        if print_result:
            print('Epoch {}'.format(epoch))
            print('Train Loss {:.4f}'.format(train_loss))
            print('Valid Loss {:.4f}'.format(valid_loss))
            print('Test Bin Acc {:.4f}'.format(test_binary_accuracy))
        
        if result['valid_loss'][-1] < min(result['valid_loss'][:-1]):
            max_accuracy = test_binary_accuracy
            model_state_dict = model.state_dict()

    numel = model.count_parameters()

    return max_accuracy, result, model_state_dict, numel

def binary_map_train_ARF(max_rank, prior_type, log_prior_coeff, learning_rate, dropout=0.15,
                         epochs=50, no_log_prior_epochs=5, eta=None, 
                         batch_size=32, device=None, dtype=torch.float32, print_result=False):

    train_set, valid_set, test_set = get_cmu_mosi_dataset(binary=True, device=device, dtype=dtype)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=len(valid_set))
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))

    input_sizes = (300, 5, 20)
    hidden_sizes = (128, 32, 32)
    fusion_size = 128
    out_size = 1
    dropouts = (dropout, dropout, dropout, dropout, dropout)
    model = ARF(input_sizes, hidden_sizes, fusion_size, max_rank, out_size, dropouts, 
                 prior_type=prior_type, eta=eta, device=device, dtype=dtype)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    '''
    fusion_parameters = list(model.fusion_layer.parameters())
    subnet_parameters = list(model.text_subnet.parameters()) + list(model.audio_subnet.parameters()) + \
        list(model.video_subnet.parameters()) + list(model.inference_subnet.parameters())
    optimizer = optim.Adam([{'params': subnet_parameters},
                           {'params': fusion_parameters, 'lr': learning_rate}], lr=5e-4)
    '''
    optimizer = optim.Adam(list(model.parameters()), learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=print_result)
        
    max_accuracy = 0

    result = dict()
    result['train_loss'] = [float('inf')]
    result['valid_loss'] = [float('inf')]
    result['test_accuracy'] = [0.0]
    result['test_f1'] = [0.0]
    result['state_dict'] = []
    
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        for text, audio, vision, label in train_dataloader:
            model.zero_grad()
            output = model(text, audio, vision)
            '''
            if epoch < no_log_prior_epochs:
                loss = criterion(output, label)
            else:
                loss = criterion(output, label) - log_prior_coeff * model.get_log_prior()
            '''
            loss = criterion(output, label) - get_log_prior_coeff(log_prior_coeff, epoch, epochs, no_log_prior_epochs) * model.get_log_prior()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss = train_loss / len(train_set)
        result['train_loss'].append(train_loss)
        
        model.eval()
        for text, audio, vision, label in valid_dataloader:
            output = model(text, audio, vision)
            valid_loss = criterion(output, label).item()
        valid_loss = valid_loss / len(valid_set)
        result['valid_loss'].append(valid_loss)
        scheduler.step(valid_loss)

        for text, audio, vision, label in test_dataloader:
            output = model(text, audio, vision)
        
        output = (output > 0).type(dtype)
        test_binary_accuracy = accuracy_score(label.cpu(), output.cpu())
        test_f1_score = f1_score(label.cpu(), output.cpu())
        result['test_accuracy'].append(test_binary_accuracy)  
        result['test_f1'].append(test_f1_score)
        
        model_state_dict = model.state_dict()
        result['state_dict'].append(model_state_dict)
    
        if result['valid_loss'][-1] < min(result['valid_loss'][:-1]):
            max_accuracy = test_binary_accuracy

        if print_result:
            print('Epoch {}'.format(epoch))
            print('Train Loss {:.4f}'.format(train_loss))
            print('Valid Loss {:.4f}'.format(valid_loss))
            print('Test Bin Acc {:.4f}'.format(test_binary_accuracy))
            print(model.fusion_layer.weight_tensor.rank_parameter.data)

    return max_accuracy, result