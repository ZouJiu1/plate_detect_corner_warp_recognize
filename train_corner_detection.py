# 路径置顶
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.getcwd())
# 导入包
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import time
from models.corner_network import cornernet

# 导入文件
# from Models.Model_for_facenet import model, optimizer_model, start_epoch, flag_train_multi_gpu
from utils.corner_dataloader import train_dataloader, test_dataloader
pwd = os.path.abspath('./')
start_epoch = 0
model = cornernet()

model_path = os.path.join(pwd, 'models', 'corner')
if not os.path.exists(model_path):
    os.makedirs(model_path)
x = [int(i.split('_')[2]) for i in os.listdir(model_path) if ('corner' in i) and ('py' not in i)]
x.sort()
if len(x)==0:
    model_pathi = ''
for i in os.listdir(model_path):
    if (len(x)!=0) and ('epoch_'+str(x[-1]) in i) and ('corner' in i):
        model_pathi = os.path.join(model_path, i)
        break
    else:
        model_pathi=''

if os.path.exists(model_pathi) and ('corner' in model_pathi):
    model_state = torch.load(model_pathi)
    model.load_state_dict(model_state['model_state_dict'])
    start_epoch = model_state['epoch']
    print('loaded %s' % model_pathi)

    # model_state = torch.load(model_pathi)
    # # model.load_state_dict(model_state['model_state_dict'])
    # start_epoch = model_state['epoch']
    #
    # now_state_dict = model.state_dict()
    # state_dict = {k: v for k, v in model_state.items() if (k in now_state_dict.keys()) and \
    #               ('fc.weight' not in now_state_dict.keys())}
    # now_state_dict.update(state_dict)
    # # now_state_dict.update(pretrained_state_dict)
    # model.load_state_dict(now_state_dict)
    # print('loaded %s' % model_pathi)
else:
    print('不存在预训练模型！')

flag_train_gpu = torch.cuda.is_available()
flag_train_multi_gpu = False
if flag_train_gpu and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.cuda()
    flag_train_multi_gpu = True
    print('Using multi-gpu training.')
elif flag_train_gpu and torch.cuda.device_count() == 1:
    model.cuda()
    print('Using single-gpu training.')

def adjust_learning_rate(optimizer, epoch):
    if epoch<30:
        lr =  0.001
    elif (epoch>=30) and (epoch<60):
        lr = 0.0006
    elif (epoch >= 60) and (epoch < 90):
        lr = 0.0001
    elif (epoch>=90) and (epoch<120):
        lr = 0.00006
    elif (epoch>=160) and (epoch<190):
        lr = 0.00001
    else:
        lr = 0.000001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_optimizer(model, new_lr):
    # setup optimizer
    optimizer = 'adam'
    if optimizer == "sgd":
        optimizer_model = torch.optim.SGD(model.parameters(), lr = new_lr,
                                          momentum=0.9, dampening=0.9,
                                          weight_decay=0)
    elif optimizer == "adagrad":
        optimizer_model = torch.optim.Adagrad(model.parameters(), lr = new_lr,
                                              lr_decay=1e-4,
                                              weight_decay=0)

    elif optimizer == "rmsprop":
        optimizer_model = torch.optim.RMSprop(model.parameters(), lr = new_lr)

    elif optimizer == "adam":
        optimizer_model = torch.optim.Adam(model.parameters(), lr = new_lr,
                                           weight_decay=0)
    return optimizer_model

# 随机种子
seed = 0
optimizer_model = create_optimizer(model, 0.001)
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# 打卡时间、epoch
total_time_start = time.time()
MSE = nn.MSELoss()
# MSE = nn.L1Loss()
end_epoch = 300
# 导入l2计算的
best_roc_auc = -1
best_accuracy = -1
print('Countdown 3 seconds')
time.sleep(1)
print('Countdown 2 seconds')
time.sleep(1)
print('Countdown 1 seconds')
time.sleep(1)

# epoch大循环
for epoch in range(start_epoch, end_epoch):
    print("\ntraining on TrainDataset! ... epoch: ", epoch)
    epoch_time_start = time.time()
    triplet_loss_sum = 0
    attention_loss_sum = 0
    num_hard = 0

    model.train()  # 训练模式
    # step小循环
    progress_bar = enumerate(tqdm(train_dataloader))
    los = []
    for batch_idx, (img, coords) in progress_bar:
        img = img.cuda()
        coords = coords.cuda()
        # 模型运算
        predcoord = model(img)
        LOSS = MSE(predcoord, coords)
        los.append(LOSS.detach().cpu().numpy())
        # print(predcoord)
        # print(coords)
        # 反向传播过程
        optimizer_model.zero_grad()
        LOSS.backward()
        optimizer_model.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer_model, epoch)

    #验证集准确度
    print("Validating on TestDataset! ...")
    model.eval()  # 验证模式
    with torch.no_grad():  # 不传梯度了
        distances, labels = [], []
        progress_bar = enumerate(tqdm(test_dataloader))
        w= 0
        val = []
        for batch_index, (img_, coord_t) in progress_bar:
            img_ = img_.cuda()
            coord_t = coord_t.cuda()
            output_ = model(img_)
            if w == 0:
                print('\n', list(output_[1, :].detach().cpu().numpy()), list(coord_t[1,:].detach().cpu().numpy()))
                w += 1
            valoss = MSE(output_, coord_t).detach().cpu().numpy()
            val.append(valoss)
    print('mean loss epoch{}: {:.6f}'.format(epoch, np.mean(los)), ', val loss: {:.6f}'.format(np.mean(val)))
    # 保存模型权重
    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_model_state_dict': optimizer_model.state_dict()
    }
    #
    if flag_train_multi_gpu:
        state['model_state_dict'] = model.module.state_dict()
        # For storing best euclidean distance threshold during LFW validation
        # if flag_validate_lfw:
        # state['best_distance_threshold'] = np.mean(best_distances)
        #
    torch.save(state, 'models/corner/corner_epoch_{}_valoss_{:.6f}.pt'.format(epoch + 1, valoss))

# Training loop end
total_time_end = time.time()
total_time_elapsed = total_time_end - total_time_start
print("\nTraining finished: total time elapsed: {:.2f} hours.".format(total_time_elapsed / 3600))
