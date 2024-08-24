import os.path
import torch
from torch import nn
from torch import optim
from torch import amp
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from datetime import datetime
from utils import DataLoaderX, accuracy, param_avg_logger

import model
import train_config



def main():
    print("-"*20 + f"Train process initializing: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "-" * 20)
    device = torch.device(train_config.device)
    cudnn.benchmark = True
    scaler = amp.GradScaler()

    dataloader_train, dataloader_val = load_datasets()
    vgg_model = define_model(device)
    optimizer = define_optimizer(vgg_model)
    criterion = define_loss(device)
    lr_schedular = define_scheduler(optimizer)

    if train_config.pretrained_model_weights_path:
        pass
    else:
        print("Pretrained model weights not found.")

    if train_config.resume_model_weights_path:
        pass
    else:
        print("Resume training model not found. Start training from scratch.")

    output_dir = os.path.join('output',train_config.exp_name)
    os.makedirs(output_dir,exist_ok=True)

    tbwriter = SummaryWriter(os.path.join(output_dir,"tblogs"))
    print("Tensorboard summary writer created.")

    print("-" * 20 + "Training Start" + "-" * 20)
    for epoch in range(train_config.epochs):
        train(
            vgg_model,
            dataloader_train,
            dataloader_val,
            optimizer,
            criterion,
            epoch,
            scaler,
            tbwriter,
            device
        )
        lr_schedular.step()

        checkpoint_path = os.path.join(output_dir,"checkpoints",f"states_epoch{epoch+1}.pkl")
        state = {
            "epoch": epoch+1,
            "optimizer": optimizer.state_dict(),
            "model": vgg_model.state_dict(),
            "scheduler": lr_schedular.state_dict(),
        }
        torch.save(state,checkpoint_path)

def load_datasets(
        train_img_dir: str = train_config.train_img_dir,
        val_img_dir: str = train_config.val_img_dir,
        resized_img_size: int = train_config.resized_img_size,
        crop_img_size: int = train_config.crop_img_size,
        dataset_mean_normalize=train_config.dataset_mean_normalize,
        dataset_std_normalize=train_config.dataset_std_normalize,
) -> [DataLoaderX, DataLoaderX]:
    transforms_train = transforms.Compose([
        transforms.Resize(resized_img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(crop_img_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean_normalize, dataset_std_normalize)
    ])
    transforms_val = transforms.Compose([
        transforms.Resize(resized_img_size),
        transforms.RandomCrop(crop_img_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean_normalize, dataset_std_normalize)

    ])

    dataset_train = ImageFolder(train_img_dir, transform=transforms_train)
    dataset_val = ImageFolder(val_img_dir, transforms_val)
    print("Datasets created.")

    dataloader_train = DataLoaderX(
        dataset_train,
        shuffle=True,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True)

    dataloader_val = DataLoaderX(
        dataset_val,
        shuffle=True,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True)
    print("Dataloaders created.")

    return dataloader_train, dataloader_val

def define_model(
        device:torch.device,
        model_arch_name:str = train_config.model_arch_name,
        model_num_classes:int = train_config.model_num_classes,
) -> nn.Module:
    vgg_model:nn.Module = model.__dict__[model_arch_name](num_classes = model_num_classes)
    vgg_model = vgg_model.to(device)
    vgg_model = nn.parallel.DataParallel(vgg_model,device_ids=train_config.device_id)
    print(f"Model created: {model_arch_name}.")
    print('-' * 20 + "Model Info" + "-" * 20)
    print(vgg_model)
    print("-" * 50)
    return vgg_model

def define_optimizer(
        model:nn.Module,
        lr:float = train_config.optim_lr,
        momentum:float = train_config.optim_momentum,
        weight_decay = train_config.optim_weight_decay
) -> optim.Optimizer:
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    print("Optimizer created.")

    return optimizer

def define_loss(
        device:torch.device,
        loss_label_smoothing:float = train_config.loss_label_smoothing,
) -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing = loss_label_smoothing)
    criterion = criterion.to(device)
    print("Criterion created.")

    return criterion


def define_scheduler(
        optimizer: optim.Optimizer,
        step_size:int = train_config.sched_step,
        gamma:float = train_config.sched_gamma
) -> optim.lr_scheduler.LRScheduler:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size,gamma)
    print("Lr_scheduler created.")

    return lr_scheduler

def train(
        model:nn.Module,
        dataloader_train:DataLoaderX,
        dataloader_val:DataLoaderX,
        optimizer:optim.Optimizer,
        criterion:nn.CrossEntropyLoss,
        epoch:int,
        scaler:amp.GradScaler,
        tbwriter:SummaryWriter,
        device:torch.device,
) -> None:
    model.train()
    step = 1
    for imgs, classes in dataloader_train:
        imgs, classes = imgs.to(device), classes.to(device)

        model.zero_grad(set_to_none=True)

        with amp.autocast(device_type=train_config.device):
            output = model(imgs)
            loss = criterion(output,classes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % train_config.train_print_frequency == 0:
            with torch.no_grad():
                top1, top5 = accuracy(output,classes,topk=(1,5))
                print(f"Epoch: {epoch + 1}/{train_config.epochs} \tStep: {step} \tLoss: {loss.item():.4f} \tAcc@1: {top1[0]:.4f}% \tAcc@5: {top5[0]:.4f}%")
                tbwriter.add_scalar('loss',loss.item(),step)
                tbwriter.add_scalar('acc@1',top1[0],step)
                tbwriter.add_scalar('acc@5', top5[0], step)

                param_avg_logger(model)

        if step % train_config.val_print_frequency == 0:
            with torch.no_grad():
                model.eval()
                val_cLoss = 0
                val_cTop1 = 0
                val_cTop5 = 0
                val_count = 0

                for val_imgs, val_classes in dataloader_val:
                    val_imgs, val_classes = val_imgs.to(device), val_classes.to(device)
                    with amp.autocast(device_type=train_config.device):
                        val_output = model(val_imgs)
                        val_cLoss += criterion(val_output,val_classes)
                    val_acc = accuracy(val_output,val_classes,topk=(1,5))
                    val_cTop1 += val_acc[0][0]
                    val_cTop5 += val_acc[1][0]

                    val_count += 1
                val_loss = val_cLoss / val_count
                val_top1 = val_cTop1 / val_count
                val_top5 = val_cTop5 / val_count

                print(f"Epoch: {epoch + 1}/{train_config.epochs} \tVal_Loss: {val_loss:.4f} \tVal_Acc@1: {val_top1:.4f}% \tVal_Acc@5: {val_top5:.4f}%")
                tbwriter.add_scalar('val_loss', val_loss, step)
                tbwriter.add_scalar('val_acc@1', val_top1, step)
                tbwriter.add_scalar('val_acc@5', val_top5, step)
        step += 1


if __name__ == '__main__':
    main()

