import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchvision.models import resnet18
import timm
from torch.utils.data import DataLoader
from skimage.io import imread
import sklearn
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import random
from early_stopping import EarlyStopping
import os
from airogs_dataset import Airogs
import wandb
import sys
import yaml
torch.multiprocessing.set_sharing_strategy('file_system')

def main(path_to_config,folds=5):
    with open(path_to_config,'r') as file_:
        config = yaml.safe_load(file_)

    fold_num = config['fold_num']
    exp_id = config['exp_id']
    resize = config['resize']
    epochs = config['epochs']
    lr = float(config['lr'])
    lr_step_period = config['lr_step_period']
    momentum = config['momentum']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    data_dir = config['data_dir']
    images_dir_name = config['images_dir_name']
    run_test = config['run_test']
    pretrained = config['pretrained']
    model_name = config['model_name']
    optimizer_name = config['optimizer_name']
    scheduler = config['scheduler']
    patience =  config['patience']
    apply_augs = config['apply_augs']
    apply_clahe = config['apply_clahe']
    dropout = config['dropout']
    try:
        apply_scaling = config['apply_scaling']
    except:
        apply_scaling = False

    try:
        polar_transform = config['polar_transform']
    except:
        polar_transform = False


    output_dir = f"output/{exp_id}_{fold_num}"
    assert fold_num in range(5), "Fold number has to be betwen 0-4."
    assert not os.path.exists(output_dir), 'Fold already exists!'
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu") 

    transform = None

    if resize != None and resize != 512:
        print(f"Using resized image size {resize}x{resize}")
        transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
        ])
        augs = [            
            transforms.ToTensor(),
            transforms.Resize(resize),
        ]
        if apply_augs:
            augs.append(transforms.RandomVerticalFlip())
            augs.append(transforms.RandomHorizontalFlip())
            augs.append(transforms.RandomRotation(10))
        
        train_transform = torchvision.transforms.Compose(augs)

    else:
        print('Using original image size 512x512')
        transform = torchvision.transforms.Compose([
            transforms.ToTensor()
        ])
        augs = [
            transforms.ToTensor(),
        ]
        if apply_augs:
            augs.append(transforms.RandomVerticalFlip())
            augs.append(transforms.RandomHorizontalFlip())
            augs.append(transforms.RandomRotation(10))
        if apply_scaling:
            augs.append(transforms.RandomAffine(0,scale=(1,1.5)))
        train_transform = torchvision.transforms.Compose(augs)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    f1s, aucs, accuracies, losses = [], [], [], []
    
    for k in range(folds):
        if k != fold_num:
            continue

        name = f"exp{exp_id}_{k}fold_{model_name}_{resize}"
        os.environ["WANDB_RUN_GROUP"] = f"exp{exp_id}_{model_name}_{resize}"
        wandb.init(project="airogs_final", entity="airogs", name=name, reinit=True)

        train_dataset = Airogs(
            path=data_dir,
            images_dir_name=images_dir_name,
            split=f"train_{k}",
            transforms=train_transform,
            polar_transforms=polar_transform,
            apply_clahe = apply_clahe
        )
        val_dataset = Airogs(
            path=data_dir,
            images_dir_name=images_dir_name,
            split=f"val_{k}",
            transforms=transform,
            polar_transforms=polar_transform,
            apply_clahe = apply_clahe
        )

        train_loader = DataLoader(train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
        )

        csv_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
        labels_referable = csv_data['referable']
        weight_referable = class_weight.compute_class_weight(class_weight='balanced', classes = np.unique(labels_referable), y=labels_referable).astype('float32')
        print("Class Weights: ", weight_referable)

        wandb.config.update ({
            "epochs": epochs,
            "lr": lr,
            "lr_step_period": lr_step_period,
            "momentun": momentum,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "data_dir": data_dir,
            "images_dir_name": images_dir_name,
            "output_dir": output_dir,
            "run_test": run_test,
            "pretrained": pretrained,
            "model": model_name,
            "optimizer": optimizer_name,
            "device": device.type,
            "resize": resize,
            "train_count": len(train_dataset),
            "val_count": len(val_dataset),
            "patience" : patience,
            "scheduler": scheduler,
            "class_weights": ", ".join(map(lambda x: str(x), weight_referable)),
            "apply_augs": apply_augs,
            "apply_clahe": apply_clahe,
            "dropout": dropout,
            "fold_num": fold_num,
            "apply_scaling": apply_scaling,
            "polar_transform": polar_transform
        })

        if scheduler == None:
            print("Scheduler is not set")


        if model_name in timm.list_models(model_name):
            model = timm.create_model(model_name,pretrained=pretrained,num_classes=2,drop_rate=dropout)
        else:
            assert False, f"Model {model_name} not  recognized"
        model = model.to(device)

        print(f"Using Model: {model_name}")

        wandb.watch(model)

        criterion = CrossEntropyLoss(weight=torch.from_numpy(weight_referable).to(device))

        if optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(),lr=lr)
        else:
            assert False, f"Optimizer {optimizer} not  recognized"

        if lr_step_period is None:
            lr_step_period = math.inf
        
        if scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_period)
        elif scheduler == 'cosine':
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

        with open(os.path.join(output_dir, f"log_{exp_id}_{fold_num}.csv"), "a") as f:
            f.write("Train Dataset size: {}".format(len(train_dataset)))
            f.write("Validation Dataset size: {}".format(len(val_dataset)))

            epoch_resume = 0
            best_f1 = 0.0
            try:
                # Attempt to load checkpoint
                checkpoint = torch.load(os.path.join(output_dir, "checkpoint.pt"))
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['opt_dict'])
                if scheduler != None:
                    scheduler.load_state_dict(checkpoint['scheduler_dict'])
                epoch_resume = checkpoint["epoch"] + 1
                best_f1 = checkpoint["best_f1"]
                f.write("Resuming from epoch {}\n".format(epoch_resume))
                f.flush()
            except FileNotFoundError:
                f.write("Starting run from scratch\n")

            # Train
            if epoch_resume < epochs:
                print(f"---------\nTraining Fold {str(k)}\n")
                f.write("--------\nTraining Fold {}\n".format(str(k)))
                for epoch in range(epoch_resume, epochs):
                    for split in ['Train', 'Val']:
                        if split == "Train":
                            model.train()
                            epoch_total_loss = 0
                            labels = []
                            predictions = []
                            loader = train_loader
                            for batch_num, (inp, target) in enumerate(tqdm(loader)):
                                labels+=target
                                optimizer.zero_grad()
                                output = model(inp.to(device))
                                _, batch_prediction = torch.max(output, dim=1)
                                predictions += batch_prediction.detach().tolist()
                                batch_loss = criterion(output, target.to(device))
                                epoch_total_loss += batch_loss.item()
                                batch_loss.backward()
                                optimizer.step()
                        else:
                            model.eval()
                            with torch.no_grad():
                                epoch_total_loss = 0
                                labels = []
                                predictions = []
                                loader = val_loader
                                for batch_num, (inp, target) in enumerate(tqdm(loader)):
                                    labels+=target
                                    output = model(inp.to(device))
                                    _, batch_prediction = torch.max(output, dim=1)
                                    predictions += batch_prediction.detach().tolist()
                                    batch_loss = criterion(output, target.to(device))
                                    epoch_total_loss += batch_loss.item()

                        avrg_loss = epoch_total_loss / loader.dataset.__len__()
                        accuracy = metrics.accuracy_score(labels, predictions)
                        confusion = metrics.confusion_matrix(labels, predictions)
                        _f1_score = f1_score(labels, predictions, average="macro")
                        auc = sklearn.metrics.roc_auc_score(labels, predictions)
                        print("%s Epoch %d - loss=%0.4f AUC=%0.4f F1=%0.4f  Accuracy=%0.4f" % (split, epoch, avrg_loss, auc, _f1_score, accuracy))
                        f.write("{} Epoch {} - loss={} AUC={} F1={} Accuracy={}\n".format(split, epoch, avrg_loss, auc, _f1_score, accuracy))
                        f.flush()
                        
                        if split == 'Train':
                            wandb.log({"epoch": epoch, "train loss": avrg_loss, "train acc": accuracy, "train f1": _f1_score, "train auc": auc})
                        else:
                            wandb.log({"epoch": epoch, "val loss": avrg_loss, "val acc": accuracy, "val f1": _f1_score, "val auc": auc})
                            f1s.append(_f1_score)
                            aucs.append(auc)
                            accuracies.append(accuracy)
                            losses.append(avrg_loss)
                            
                            early_stopping(avrg_loss, model)
                            if early_stopping.early_stop:
                                print("Early stopping")
                                break



                        if k == (folds - 1) and split == "Val":
                            f1_mean = np.mean(f1s)
                            auc_mean = np.mean(aucs)
                            acc_mean = np.mean(accuracies)
                            loss_mean = np.mean(losses)

                            print("--------\n%s Epoch %d - mean_loss=%0.4f mean_AUC=%0.4f mean_F1=%0.4f  mean_Accuracy=%0.4f" % (split, epoch, loss_mean, auc_mean, f1_mean, acc_mean))
                            f.write("--------\n{} Epoch {} - mean_loss={} mean_AUC={} mean_F1={} mean_Accuracy={}\n".format(split, epoch, loss_mean, auc_mean, f1_mean, acc_mean))
                            wandb.log({"epoch": epoch, "val mean loss": loss_mean, "val mean acc": acc_mean, "val mean f1": f1_mean, "val mean auc": auc_mean})
                            f.flush()

                    if scheduler != None:
                        scheduler.step()

                    # save model
                    checkpoint = {
                        'epoch': epoch,
                        'best_f1': best_f1,
                        'f1': _f1_score,
                        'auc': auc,
                        'loss': avrg_loss,
                        'state_dict': model.state_dict(),
                        'opt_dict': optimizer.state_dict()
                    }

                    if scheduler != None:
                        checkpoint["scheduler_dict"] = scheduler.state_dict()

                    torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_{fold_num}.pt"))
                    if _f1_score > best_f1:
                        best_f1 = _f1_score
                        checkpoint["best_f1"] = best_f1
                        torch.save(checkpoint, os.path.join(output_dir, f"best_{fold_num}.pt"))

            else:
                print("Skipping training\n")
                f.write("Skipping training\n")
                f.flush()

            # Testing
            if run_test:
                with open(os.path.join(output_dir, f"log_{exp_id}_{fold_num}.csv"), "a") as f: 
                    # Best F1 Score
                    # max_f1 = max(f1s)
                    # max_index = f1s.index(max_f1)

                    checkpoint = torch.load(os.path.join(output_dir, f"best_{fold_num}.pt"))
                    model.load_state_dict(checkpoint['state_dict'])
                    f.write("---------\nBest F1 {} for fold {}  epoch {}\n".format(checkpoint["best_f1"], str(fold_num), checkpoint["epoch"]))
                    f.flush()
                    print("-----------\nBest F1 {} for fold {} epoch {}\n".format(checkpoint["best_f1"], str(fold_num), checkpoint["epoch"]))

                    test_dataset = Airogs(
                        path=data_dir,
                        images_dir_name=images_dir_name,
                        split="test",
                        transforms=transform,
                        polar_transforms=polar_transform,
                        apply_clahe = apply_clahe
                    )
                    test_loader = DataLoader(test_dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers=num_workers,
                    )


                    model.eval()
                    labels = []
                    predictions = []
                    with torch.no_grad():
                        for (inp, target) in tqdm(test_loader):
                            labels+=target
                            batch_prediction = model(inp.to(device))
                            _, batch_prediction = torch.max(batch_prediction, dim=1)
                            predictions += batch_prediction.detach().tolist()
                    accuracy = metrics.accuracy_score(labels, predictions)
                    f.write("Test Accuracy = {}\n".format(accuracy))
                    print("Test Accuracy = %0.2f" % (accuracy))
                    _f1_score = f1_score(labels, predictions, average="macro")
                    f.write("Test F1 Score = {}\n".format(_f1_score))
                    print("Test F1 = %0.2f" % (_f1_score))
                    auc = sklearn.metrics.roc_auc_score(labels, predictions)
                    f.write("Test AUC = {}\n".format(auc))
                    print("Test AUC = %0.2f" % (auc))
                    confusion = metrics.confusion_matrix(labels, predictions)
                    f.write("Confusion Matrix = {}\n".format(confusion))
                    print(confusion)
                    f.flush()

                    wandb.log({"test acc": accuracy, "test f1": _f1_score, "test auc": auc})



if __name__ == "__main__":
    path_to_config = sys.argv[1]
    main(path_to_config)
