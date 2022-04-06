from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

import time
from .dataset import get_cmu_mosi_dataset
from .net import *
from .util import get_log_prior_coeff

def binary_map_train_TFN(model, train_set, test_set, learning_rate, epochs=50, batch_size=32, 
                         print_result=False):

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    #optimizer = optim.Adam(list(model.parameters()), learning_rate, weight_decay=0.01)
    optimizer = optim.Adam(list(model.parameters()), learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=print_result)
    
    max_accuracy = 0

    result = dict()
    result['train_loss'] = [float('inf')]
    result['valid_loss'] = [float('inf')]
    result['test_accuracy'] = [0.0]
    result['test_f1'] = [0.0]
    result['train_time'] = []

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        tic = time.time()
        for text, audio, vision, label in train_dataloader:
            model.zero_grad()
            output = model(text, audio, vision)
            loss = criterion(output, label)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        toc = time.time()
        result['train_time'].append(toc - tic)
        train_loss = train_loss / len(train_set)
        result['train_loss'].append(train_loss)
        
        model.eval()
        for text, audio, vision, label in test_dataloader:
            output = model(text, audio, vision)
            valid_loss = criterion(output, label).item()
        valid_loss = valid_loss / len(test_set)
        result['valid_loss'].append(valid_loss)
        scheduler.step(valid_loss)
        
        output = (output > 0).type(output.dtype)

        test_binary_accuracy = accuracy_score(label.cpu(), output.cpu())
        test_f1_score = f1_score(label.cpu(), output.cpu())
        result['test_accuracy'].append(test_binary_accuracy)  
        result['test_f1'].append(test_f1_score)
        
        if result['valid_loss'][-1] < min(result['valid_loss'][:-1]):
            max_accuracy = test_binary_accuracy

        if print_result:
            print('Epoch {}'.format(epoch))
            print('Train Loss {:.4f}'.format(train_loss))
            print('Valid Loss {:.4f}'.format(valid_loss))
            print('Test Bin Acc {:.4f}'.format(test_binary_accuracy))

    return max_accuracy, result 


def binary_map_train_LMF(model, train_set, test_set, learning_rate, epochs=50, batch_size=32, 
                         print_result=False):

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(list(model.parameters()), learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=print_result)
    
    max_accuracy = 0

    result = dict()
    result['train_loss'] = [float('inf')]
    result['valid_loss'] = [float('inf')]
    result['test_accuracy'] = [0.0]
    result['test_f1'] = [0.0]
    result['train_time'] = []

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        
        tic = time.time()
        for text, audio, vision, label in train_dataloader:
            model.zero_grad()
            output = model(text, audio, vision)
            loss = criterion(output, label)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        toc = time.time()
        train_loss = train_loss / len(train_set)
        result['train_loss'].append(train_loss)
        result['train_time'].append(toc-tic)
        
        model.eval()
        for text, audio, vision, label in test_dataloader:
            output = model(text, audio, vision)
            valid_loss = criterion(output, label).item()
        valid_loss = valid_loss / len(test_set)
        result['valid_loss'].append(valid_loss)
        scheduler.step(valid_loss)
        
        output = (output > 0).type(output.dtype)

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

    return max_accuracy, result

def binary_map_train_ARF(model, train_set, test_set, log_prior_coeff, learning_rate,
                         epochs=50, batch_size=32, print_result=False):

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))

    criterion = nn.BCEWithLogitsLoss()
    
    fusion_parameters = list(model.fusion_layer.parameters())
    subnet_parameters = list(model.text_subnet.parameters()) + list(model.audio_subnet.parameters()) + \
        list(model.video_subnet.parameters()) + list(model.inference_subnet.parameters())
    optimizer = optim.Adam([{'params': subnet_parameters},
                           {'params': fusion_parameters, 'lr': learning_rate}], lr=5e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=print_result)
        
    max_accuracy = 0

    result = dict()
    result['train_loss'] = [float('inf')]
    result['valid_loss'] = [float('inf')]
    result['test_accuracy'] = [0.0]
    result['test_f1'] = [0.0]
    result['rank'] = []
    result['train_time'] = []
    result['norms'] = []
    result['grad_norm'] = []
    result['grad_std'] = []

    max_rank = model.fusion_layer.max_rank
    
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        tic = time.time()
        for text, audio, vision, label in train_dataloader:
            model.zero_grad()
            output = model(text, audio, vision)
            loss = criterion(output, label) - log_prior_coeff * model.get_log_prior()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss = train_loss / len(train_dataloader)
        result['train_loss'].append(train_loss)
        toc = time.time()
        result['train_time'].append(toc-tic)

        model.eval()

        for text, audio, vision, label in test_dataloader:
            output = model(text, audio, vision)
            valid_loss = criterion(output, label).item()
        result['valid_loss'].append(valid_loss)
        scheduler.step(valid_loss)
        
        output = (output > 0).type(output.dtype)
        test_binary_accuracy = accuracy_score(label.cpu(), output.cpu())
        test_f1_score = f1_score(label.cpu(), output.cpu())
        result['test_accuracy'].append(test_binary_accuracy)  
        result['test_f1'].append(test_f1_score)
        current_rank = model.fusion_layer.estimate_rank().item()
        result['rank'].append(current_rank)
        norm = 0
        grad_norm = 0
        grad_std = 0
        for factor in model.fusion_layer.weight_tensor_factors:
            norm = norm + factor.norm(dim=0).detach().cpu().numpy()
            grad_norm = grad_norm + factor.grad.norm(dim=0).cpu().numpy()
            grad_std = grad_std + factor.grad.std(dim=0).cpu().numpy()
        norm = norm / 4
        grad_norm = grad_norm / 4
        grad_std = grad_std  / 4
        result['norms'].append(norm)
        result['grad_norm'].append(grad_norm)
        result['grad_std'].append(grad_std)
    
        if result['valid_loss'][-1] < min(result['valid_loss'][:-1]):
            max_accuracy = test_binary_accuracy

        if result['test_accuracy'][-1] > .77 and current_rank < max_rank:
            model_state_dict = model.state_dict()
            name = str(epoch) + '_state_dict.pt'
            torch.save(model_state_dict, name)

        if result['test_accuracy'][-5:] == [result['test_accuracy'][-1]] * 5:
            break
        
        if print_result:
            print('Epoch {}'.format(epoch))
            print('Train Loss {:.4f}'.format(train_loss))
            print('Valid Loss {:.4f}'.format(valid_loss))
            print('Test Bin Acc. {:.4f}'.format(test_binary_accuracy))
            print('Current Rank Est. {}'.format(current_rank))

    return max_accuracy, result

def binary_map_train_ARF_with_AR_TextSubNet(model, train_set, test_set, log_prior_coeff, learning_rate,
                         epochs=50, batch_size=32, print_result=False):

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))

    criterion = nn.BCEWithLogitsLoss()
    
    fusion_parameters = list(model.fusion_layer.parameters())
    text_subnet_parameters = list(model.text_subnet.parameters())

    other_parameters = list(model.audio_subnet.parameters()) + \
        list(model.video_subnet.parameters()) + list(model.inference_subnet.parameters())
    optimizer = optim.Adam([{'params': other_parameters},
                           {'params': fusion_parameters, 'lr': 5e-3},
                           {'params': text_subnet_parameters, 'lr': learning_rate}], 
                           lr=5e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=print_result)
        
    max_accuracy = 0

    result = dict()
    result['train_loss'] = [float('inf')]
    result['valid_loss'] = [float('inf')]
    result['test_accuracy'] = [0.0]
    result['test_f1'] = [0.0]
    result['rank'] = []
    result['train_time'] = []

    max_rank = model.fusion_layer.max_rank
    
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        tic = time.time()
        for text, audio, vision, label in train_dataloader:
            model.zero_grad()
            output = model(text, audio, vision)
            loss = criterion(output, label) - log_prior_coeff * model.get_log_prior()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss = train_loss / len(train_dataloader)
        result['train_loss'].append(train_loss)
        toc = time.time()
        result['train_time'].append(toc-tic)

        model.eval()

        for text, audio, vision, label in test_dataloader:
            output = model(text, audio, vision)
            valid_loss = criterion(output, label).item()
        result['valid_loss'].append(valid_loss)
        scheduler.step(valid_loss)
        
        output = (output > 0).type(output.dtype)
        test_binary_accuracy = accuracy_score(label.cpu(), output.cpu())
        test_f1_score = f1_score(label.cpu(), output.cpu())
        result['test_accuracy'].append(test_binary_accuracy)  
        result['test_f1'].append(test_f1_score)
        current_rank = model.fusion_layer.estimate_rank().item()
        result['rank'].append(current_rank)
    
        if result['valid_loss'][-1] < min(result['valid_loss'][:-1]):
            max_accuracy = test_binary_accuracy

        if result['test_accuracy'][-1] > .77 and current_rank < max_rank:
            model_state_dict = model.state_dict()
            name = str(epoch) + '_state_dict.pt'
            torch.save(model_state_dict, name)

        if result['test_accuracy'][-5:] == [result['test_accuracy'][-1]] * 5:
            break
        
        if print_result:
            print('Epoch {}'.format(epoch))
            print('Train Loss {:.4f}'.format(train_loss))
            print('Valid Loss {:.4f}'.format(valid_loss))
            print('Test Bin Acc. {:.4f}'.format(test_binary_accuracy))
            print('Current Rank Est. {}'.format(current_rank))
            print(model.text_subnet.rnn.layer_ih.weight_tensor.estimate_rank())
            print(model.text_subnet.rnn.layer_hh.weight_tensor.estimate_rank())
            print(model.text_subnet.linear_1.weight_tensor.estimate_rank())
            print(model.text_subnet.linear_2.weight_tensor.estimate_rank())

    return max_accuracy, result