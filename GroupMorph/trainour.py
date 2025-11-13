import os, glob, sys, datetime
from argparse import ArgumentParser
import torch.utils.data as Data
from torch.optim import lr_scheduler
from GroupMorph.Model.Loss import *
from GroupMorph.Model.Net import GruopMorph
from GroupMorph.Model.Function import Dataset_OASIS, SpatialTransformer
from getdata import Ourdata
from getourfile import getdirfile
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import json

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=160001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--dice", type=float,
                    dest="dice", default=1,
                    help="Dice loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000,
                    help="frequency of saving models")
parser.add_argument("--bs_ch", type=int,
                    dest="bs_ch", default=8,
                    help="number of basic channels")
parser.add_argument("--modelname", type=str,
                    dest="model_name",
                    default='reg',
                    help="Name for saving")
parser.add_argument("--gpu", type=str,
                    dest="gpu",
                    default='0',
                    help="gpus")
parser.add_argument("--classes", type=int,
                    dest="classes",
                    default='36',
                    help="number classes")
opt = parser.parse_args()

lr = opt.lr
bs_ch = opt.bs_ch
local_ori = opt.local_ori
n_checkpoint = opt.checkpoint
smooth = opt.smooth
dice = opt.dice
model_name = opt.model_name
iteration = opt.iteration
classes = opt.classes
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

imgshape = (32,192,320)#(160, 192, 192)
groups = (4, 2, 2)  # (4,4,4), (4,4,2), (4,2,2) or (2,2,2)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def train():
    save_dir = 'GroupMorph_Ncc'
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    savepathES = r'logs/Groupmorph_Ncc'
    if not os.path.exists(savepathES):
        os.makedirs(savepathES)
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    early_stopping = EarlyStopping(savepath=save_dir,name=r'D:\mx\CT_Re\GroupMorph\experiments\GroupMorph_Ncc', patience=20, verbose=True, )
    model = GruopMorph(1, 8, imgshape, groups).cuda()



    loss_similarity = ncc_loss
    transfor = SpatialTransformer()#.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[88000], gamma=0.1)

    model_dir = 'Model_weight/T-test1'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    loss_all = np.zeros((5, iteration + 1))

    step = 0

    load_model = False
    if load_model is True:
        model_path = ''
        step = 88000
        model.load_state_dict(torch.load(model_path))
        loss_load = np.load("")
        loss_all[:, :step] = loss_load[:, :step]

    # names = sorted(glob.glob('../neurite-oasis.v1.0/OASIS_OAS1_*_MR1'))[0:255]  # 255
    # training_generator = Data.DataLoader(Dataset_OASIS(names, norm=False), batch_size=1,
    #                                      shuffle=True, num_workers=2)
    # fixed_names = sorted(glob.glob('../neurite-oasis.v1.0/OASIS_OAS1_*_MR1'))[277:282]  # 255
    # fixed_generator = Data.DataLoader(Dataset_OASIS(fixed_names, norm=False), batch_size=1,
    #                                      shuffle=True, num_workers=2)
    sample = None
    with open(r"D:\mx\CT_Re\train_5.json", 'r') as f:
        data = json.load(f)
    traindataset = data['train']
    valdataset = getdirfile('val')
    traindata = Ourdata(traindataset,None,None)
    valdata = Ourdata(valdataset,None,None)

    traindata = Data.DataLoader(traindata, batch_size=2, shuffle=True, num_workers=2)
    valdata = Data.DataLoader(valdata, batch_size=2, shuffle=False)

    maxdice = 0
    while step <= iteration:
        model.train()
        for batch_idx, data in enumerate(traindata):
            X,Y,X_label,Y_label = data[0].cuda(),data[1].cuda(),data[2].cuda(),data[3].cuda()
            X = X.float()
            Y = Y.float()
            X_label = X_label.float()
            Y_label = Y_label.float()

            flows, warps, smo = model(X, Y)

            # dice loss
            # Y_label_onehot = mask_to_one_hot(Y_label, n_classes=classes)
            # X_label_onehot = mask_to_one_hot(X_label, n_classes=classes)
            warps_label_onehot = transfor(X_label, flows)
            diceloss = compute_per_channel_dice(warps_label_onehot, Y_label, classes=4)
            # dice loss


            sim = loss_similarity(warps, Y)
            smo_loss = smo
            loss = sim  + smooth * smo_loss
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            l = optimizer.param_groups[0]['lr']
            loss_all[:, step] = np.array(
                [loss.item(), sim.item(), diceloss.item(), smo_loss.item(), sim.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_loss "{2:4f}" - dice_loss "{3:4f}" - smo_loss "{4:4f}" - lr:"{5:.6f}"'.format(
                    step, loss.item(), sim.item(), diceloss.item(), smo_loss.item(), l))
            sys.stdout.flush()

        model.eval()
        dice_total = []
        with torch.no_grad():
            for data in valdata:
                X, Y, X_label, Y_label = data[0].cuda(),data[1].cuda(),data[2].cuda(),data[3].cuda()
                X = X.float()
                Y = Y.float()
                X_label = X_label.float()
                Y_label = Y_label.float()
                flows, warps, smo = model(X, Y)
                sim = loss_similarity(warps, Y)
                # smo_loss = smo
                X_Y_label = transfor(X_label, flows, mode='nearest')
                dice_score_reg = dicegup(X_Y_label, Y_label)
                dice_score_reg = dice_score_reg.cpu().numpy()
                dice_total.append(dice_score_reg)
            dice_total = np.array(dice_total)
            maxdice = max(dice_total.mean(),maxdice)
            print("Dice mean: ", dice_total.mean())
        early_stopping(-dice_total.mean(), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(maxdice)

    # while step <= iteration:
    #     for batch_idx, data in enumerate(training_generator):
    #         X, X_label = data['image'].cuda(), data['image_label'].cuda()
    #         for batch_idx_fixed, data_fixed in enumerate(fixed_generator):
    #             Y, Y_label = data_fixed['image'].cuda(), data_fixed['image_label'].cuda()
    #             X = X.float()
    #             Y = Y.float()
    #
    #             flows, warps, smo = model(X, Y)
    #
    #             # dice loss
    #             Y_label_onehot = mask_to_one_hot(Y_label, n_classes=classes)
    #             X_label_onehot = mask_to_one_hot(X_label, n_classes=classes)
    #             warps_label_onehot = transfor(X_label_onehot, flows)
    #             diceloss = compute_per_channel_dice(warps_label_onehot, Y_label_onehot, classes=classes)
    #             # dice loss
    #
    #             sim = loss_similarity(warps, Y)
    #             smo_loss = smo
    #             loss = sim + dice * diceloss + smooth * smo_loss
    #             optimizer.zero_grad()
    #             loss.backward()
    #
    #             optimizer.step()
    #             scheduler.step()
    #
    #             l = optimizer.param_groups[0]['lr']
    #             loss_all[:, step] = np.array(
    #                 [loss.item(), sim.item(), diceloss.item(), smo_loss.item(), sim.item()])
    #             sys.stdout.write(
    #                 "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_loss "{2:4f}" - dice_loss "{3:4f}" - smo_loss "{4:4f}" - lr:"{5:.6f}"'.format(
    #                     step, loss.item(), sim.item(), diceloss.item(), smo_loss.item(), l))
    #             sys.stdout.flush()
    #             # log_dir = "log/loss.txt"
    #             # with open(log_dir, "a") as log:
    #             #     log.write(
    #             #     "\n" + 'step "{0}" -> training loss "{1:.4f}" - sim_loss "{2:4f}" - dice_loss "{3:4f}" - smo_loss "{4:4f}" - lr:"{5:.6f}"'.format(
    #             #         step, loss.item(), sim.item(), diceloss.item(), smo_loss.item(), l))
    #
    #             step += 1
    #             if (step % n_checkpoint == 0):
    #                 # save model
    #                 modelname = model_dir + '/' + model_name + str(
    #                     step) + '.pth'
    #                 torch.save(model.state_dict(), modelname)
    #                 np.save(model_dir + '/loss' + model_name + str(step) + '.npy', loss_all)
    #                 # save model
    #                 valname = sorted(glob.glob('../neurite-oasis.v1.0/OASIS_OAS1_*_MR1'))[255:277]
    #                 valid_generator = Data.DataLoader(Dataset_OASIS(valname, norm=False),
    #                                                   batch_size=1,
    #                                                   shuffle=False, num_workers=2)
    #
    #                 dice_total = []
    #                 print("\nValiding...")
    #                 for batch_idx, data in enumerate(valid_generator):
    #                     X, X_label = data['image'].cuda(), data['image_label'].cuda()
    #                     for batch_idx_fixed, data_fixed in enumerate(fixed_generator):
    #                         Y, Y_label = data_fixed['image'].cuda(), data_fixed['image_label'].cuda()
    #                         with torch.no_grad():
    #                             X = X.float()
    #                             Y = Y.float()
    #                             flows, _, _ = model(X, Y)
    #                             X_Y_label = transfor(X_label, flows, mode='nearest')[0, 0, :, :, :]
    #                             dice_score_reg = dicegup(X_Y_label, Y_label[0, 0, :, :, :])
    #                             dice_score_reg = dice_score_reg.cpu().numpy()
    #                             dice_total.append(dice_score_reg)
    #                 dice_total = np.array(dice_total)
    #                 print("Dice mean: ", dice_total.mean())
    #                 log_dir = "log/val.txt"
    #                 with open(log_dir, "a") as log:
    #                     log.write("step:" + str(step) + "Dice mean:" + str(dice_total.mean()) + "\n")
    #             if step > iteration:
    #                 break


if __name__ == '__main__':
    start = datetime.datetime.now()
    train()
    end = datetime.datetime.now()
    print("Time used:", end - start)
