import argparse
import logging
import os
import sys
import copy
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import time 
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torchvision

root_dir = os.getenv('CS231n_ROOT_DIR')
data_dir = os.getenv('CS231n_DATA_DIR')


csv_path = os.path.join(data_dir,'train.csv')
small_csv_path = os.path.join(data_dir,'small_train.csv')
tiny_csv_path = os.path.join(data_dir,'tiny_train.csv')
dir_checkpoint = os.path.join(root_dir, 'checkpoints')

# def train_model(model,             
#               device,
#               epochs=5,
#               batch_size=1,
#               lr=0.001,
#               val_percent=0.1,
#               save_cp=True
#               ):

#     dataset = BasicDataset(csv_file=tiny_csv_path,
#                                     data_dir=data_dir)
#     n_val = int(len(dataset) * val_percent)
#     n_train = len(dataset) - n_val
#     train, val = random_split(dataset, [n_train, n_val])
#     train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
#     val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

#     dataloaders = {'train': train_loader, 'val':val_loader}

#     dataset_sizes = {'train':n_train, 'val':n_val}

#     # default `log_dir` is "runs" - we'll be more specific here
#     writer = SummaryWriter('runs/experiment_1')
#     global_step = 0

#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {lr}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_cp}
#         Device:          {device.type}
#     ''')

#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     # optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#     # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' , patience=2)
#     criterion = nn.CrossEntropyLoss()
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(epochs):
#         logging.info('Epoch {}/{}'.format(epoch, epochs - 1))
            
#         for phase in ['train', 'val']:
#             if phase == 'train':                   
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0
#             running_corrects = 0

#             for batch in dataloaders[phase]:

#                 imgs = batch['image'].to(device, dtype=torch.float32)
#                 imgs = torch.reshape(imgs,(imgs.shape[0],imgs.shape[3],imgs.shape[1],imgs.shape[2])) #  inputs.reshape [N, C, W, H]

#                 true_label = batch['isup_grade'].to(device=device, dtype=torch.long)

#                 img_grid = torchvision.utils.make_grid(imgs)
#                 writer.add_image("train_img" , img_grid)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 with torch.set_grad_enabled(phase=='train'):
#                     outputs = model(imgs)
#                     _, pred_label = torch.max(outputs, 1)

#                     loss = criterion(outputs, true_label)

#                     writer.add_scalar('Training Loss', loss.item(), global_step)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 writer.add_graph(model, imgs)

#                 running_loss += loss.item() * imgs.size(0) # batch_size 
#                 # running_corrects += torch.sum(pred_label == true_label.data) 
#                 running_corrects += torch.sum(pred_label == true_label.data)

#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#             writer.add_scalar(phase + "loss", epoch_loss, global_step)
#             writer.add_scalar(phase + "acc", epoch_acc, global_step)




#             # nn.utils.clip_grad_value_(model.parameters(), 0.1)
#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())                    
                
#             global_step += 1


#             if global_step % (len(dataset) // (10 * batch_size)) == 0:
#                 for tag, value in model.named_parameters():
#                     tag = tag.replace('.', '/')
#                     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
#                     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
#                 val_score = eval_model(model, val_loader, device)
#                 scheduler.step(val_score)
#                 writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)    

#         if save_cp:
#             try:
#                 os.mkdir(dir_checkpoint)
#                 logging.info('Created checkpoint directory')
#             except OSError:
#                 pass
#             torch.save(model.state_dict(),
#                        dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
#             logging.info(f'Checkpoint {epoch + 1} saved !')
#     writer.close()



def new_train_model(model, device, epochs=5, batch_size=1, lr=0.001, val_percent=0.1, save_cp=True):

    dataset = BasicDataset(csv_file=tiny_csv_path, data_dir=data_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    dataloaders = {'train': train_loader, 'val':val_loader}

    dataset_sizes = {'train':n_train, 'val':n_val}

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/experiment_1')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    for epoch in range(epochs):
        logging.info('Epoch {}/{}'.format(epoch, epochs - 1))

        model.train()

        epoch_loss = 0.00 
        epoch_accuracy = 0.00
        running_loss = 0.00
        running_corrects = 0.00 
        for i, batch in enumerate(train_loader) :

            imgs = batch['image'].to(device, dtype=torch.float32)
            imgs = torch.reshape(imgs,(imgs.shape[0],imgs.shape[3],imgs.shape[1],imgs.shape[2])) #  inputs.reshape [N, C, W, H]

            true_label = batch['isup_grade'].to(device=device, dtype=torch.long)

            logits = model(imgs)
            _, pred_label = torch.max(logits, 1)
            loss = criterion(logits, true_label)

            epoch_loss += loss.item()
            running_loss += loss.item() * imgs.size(0) # batch_size 
            running_corrects += torch.sum(pred_label == true_label.data)

            #tensorboard
            img_grid = torchvision.utils.make_grid(imgs)
            # writer.add_image("training Images grid" , img_grid)
            writer.add_scalar('Training Loss', loss.item(), global_step)

            logging.info('Epoch {} , Train Step {}, Training loss: {}'.format(epoch, i , loss.item()))

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()


            global_step += 1
            if global_step % (len(dataset) // (10 * batch_size)) == 0:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                val_score = eval_model(model, val_loader, device)
                scheduler.step()
                logging.info('Validation loss: {}'.format(val_score))

                #tensorboard
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Validation Loss', val_score, global_step)
                writer.add_images('images', imgs, global_step)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        logging.info(' epoch {}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        writer.add_scalar("epoch loss", epoch_loss, epoch)
        writer.add_scalar("epoch acc", epoch_acc, epoch)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()



def images_to_probs(model, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]      



def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target isup_grade',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

def eval_model(model, val_loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    model.eval()

    n_val = len(val_loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in val_loader:
            imgs, true_label = batch['image'], batch['isup_grade']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_label = true_label.to(device=device, dtype=torch.long)

            imgs = torch.reshape(imgs,(imgs.shape[0],imgs.shape[3],imgs.shape[1],imgs.shape[2])) #  inputs.reshape [N, C, W, H]
            
            with torch.no_grad():
                pred_label = model(imgs)

            tot += F.cross_entropy(pred_label, true_label).item()
            pbar.update()

    model.train()
    return tot / n_val


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # net = UNet(n_channels=3, n_classes=1, bilinear=True)

    model = models.resnet50(pretrained=True).to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6) # num_classes = 6
    model_ft = model.to(device)


    if args.load:
        model.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        new_train_model(model=model_ft,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)