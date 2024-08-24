from torch.xpu import device

device = "cuda"
device_id = [0,1,2,3]

model_arch_name = 'vgg11'
model_num_classes = 1000

# experiment name, easy to save weights and log files
exp_name = "VGG11-ImageNet-1k"

train_img_dir = '/home/zhxue/projects/CNN_Impl/data/imagenet/2012/ILSVRC2012_img_train'
val_img_dir = '/home/zhxue/projects/CNN_Impl/data/imagenet/2012/ILSVRC2012_img_val'

dataset_mean_normalize = (0.485, 0.456, 0.406)
dataset_std_normalize = (0.229, 0.224, 0.225)

resized_img_size = 256
crop_img_size = 224
batch_size = 256
num_workers = 16

pretrained_model_weights_path = None
resume_model_weights_path = None

epochs = 90

optim_lr = 0.1
optim_momentum = 0.9
optim_weight_decay = 2e-5

loss_label_smoothing = 0.1

sched_step = 30
sched_gamma = 0.1

train_print_frequency = 10
val_print_frequency =1000
