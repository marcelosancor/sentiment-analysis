import torch
import sys
import math
import time
import json 
import os
import pickle
import pandas as pd

from torch import nn
from model import SentimentAnalysis
from torch import optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SADataset(Dataset):
    def __init__(self, path):
        data = pd.read_csv('data/train.csv', index_col=0).values
        self.data = torch.from_numpy(data[:, :-1])
        self.labels = torch.from_numpy(data[:, -1])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]

def print_progress(prog):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*prog, 5*prog))
    sys.stdout.flush()

def evaluate(model, loader, criterion, batches):
    model.eval()
    iterator = iter(loader)
    avg_loss = 0
    avg_acc = 0
    n=0
    for i in range(batches):
        inpts, labels = next(iterator)
        inpts = inpts.to(device)
        labels = labels.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            outputs = model(inpts)
            loss = criterion(outputs.flatten(), labels)
        avg_loss += loss.item()
        avg_acc += sum((outputs.flatten()>0.5) == labels).item()
        n += inpts.shape[0]
    return avg_loss/n, avg_acc/n
    

def train(train_loader, val_loader, model, criterion, optimizer, tconfig):
    if tconfig["load"]:
        with open(os.path.join(tconfig["path"], "model.pth"), "rb") as f:
            state_dict = torch.load(f)
        with open(os.path.join(tconfig["model_state_path"], "model_state.json"), "r") as f:
            model_state = json.load(f)
        epoch = best_epoch = model_state["best_epoch"]+1
        tloss, vloss = model_state["tloss"][:epoch-1], model_state["vloss"][:epoch-1]
        tacc, vacc = model_state["tacc"][:epoch-1], model_state["vacc"][:epoch-1]
        best_loss=vloss[epoch-2]
        model.load_state_dict(state_dict)
    else:
        epoch=best_epoch=0
        tloss, vloss = [], []
        tacc, vacc = [] ,[]
        best_loss=math.inf
    
    model_state = {
                "best_epoch": best_epoch,
                "epoch": epoch,
                "tloss": tloss,
                "vloss": vloss,
                "tacc": tacc, 
                "vacc": vacc
            }
    
    batches = tconfig["eval_size"]//tconfig["batch_size"]
    
    start = time.time()
    for i in range(epoch, tconfig["epochs"]+epoch):
        print('---------Epoch ', i, '/', tconfig["epochs"],'----------')
        model.train()
        start_epoch = time.time()
        prog=0
        for j, (inpts, labels) in enumerate(train_loader):
            inpts = inpts.to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            outputs = model(inpts)
            loss = criterion(outputs.flatten(), labels)

            if int(20*(j/len(train_loader))) > prog or j==len(train_loader)-1:
                prog = int(20*((j+1)/len(train_loader)))
                print_progress(prog)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train, acc_train = evaluate(model, train_loader, criterion, batches)
        loss_test, acc_test = evaluate(model, val_loader, criterion, batches)
        model_state["tloss"].append(loss_train)
        model_state["vloss"].append(loss_test)
        model_state["tacc"].append(acc_train)
        model_state["vacc"].append(acc_test)
        model_state["epoch"]=i

        if loss_test < best_loss:
            with open(os.path.join(tconfig["path"], "model.pth"), "wb") as f:
                torch.save(model.state_dict(), f)
            model_state["best_epoch"]=i
            best_loss = loss_test

        with open(os.path.join(tconfig["path"], "model_state.json"), "w") as f:
            json.dump(model_state, f)

        print()
        print('TRAIN LOSS: {:.6f} || TEST LOSS {:.6f}'.format(loss_train, loss_test))
        print('TRAIN ACC:  {:.3f} || TEST ACC  {:.3f}'.format(acc_train*100, acc_test*100))
        print('TIME EPOCH : {:.3f}s || TOTAL TIME: {:.3f}s'.format(time.time()-start_epoch, time.time() - start))
    
if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    tconfig = config["TRAINING"]
    assert(os.path.isdir(tconfig["path"]))
    
    if tconfig["load"]:
        with open(os.path.join(tconfig["path"], "model_config.json"), "r") as f:
            mconfig = json.load(f)
    else:    
        m = config["TRAINING"]["model_config"]
        mconfig = config["MODEL"][m]
        with open(os.path.join(tconfig["path"], "model_config.json"), "w") as f:
            json.dump(mconfig, f)

    with open(mconfig["vocab_path"], "rb") as f:
        vocab = pickle.load(f)

    model = SentimentAnalysis(
        vocab=vocab,
        out_channels=mconfig["out_channels"],
        n_blocks=mconfig["n_blocks"],
        hidden_dim=mconfig["hidden_dim"],
        num_layers=mconfig["num_layers"],
        dropout=mconfig["dropout"],
        bidirectional=mconfig["bidirectional"],
        linear_dim=mconfig["linear_dim"]
    )
    model = model.to(device)
    summary(model, input_data=torch.randint(0, len(vocab["itos"]), (tconfig["batch_size"], config["DATASET"]["window_size"])))

    criterion = BCELoss()
    if tconfig["optimizer"]=="Adam":
        optimizer = optim.Adam(lr=tconfig["learning_rate"], params=model.parameters(), weight_decay=tconfig["weight_decay"])
    else:   
        optimizer = optim.SGD(lr=tconfig["learning_rate"], params=model.parameters(), weight_decay=tconfig["weight_decay"])

    data_train = SADataset(os.path.join(config["DATASET"]["path_to_save"], "train.csv"))
    data_test = SADataset(os.path.join(config["DATASET"]["path_to_save"], "test.csv"))
    train_loader = DataLoader(data_train, batch_size=tconfig["batch_size"], shuffle=True)
    val_loader = DataLoader(data_test, batch_size=tconfig["batch_size"], shuffle=False)
    
    train(train_loader, val_loader, model, criterion, optimizer, tconfig)

    
    

