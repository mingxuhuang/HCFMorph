import os, glob, sys, datetime
from argparse import ArgumentParser
import torch.utils.data as Data
from torch.optim import lr_scheduler
from GroupMorph.Model.Loss import *
from GroupMorph.Model.Net import GruopMorph
from GroupMorph.Model.Function import Dataset_OASIS, SpatialTransformer
from getdata import Ourdata_test
from getourfile import getdirfile
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import json
import SimpleITK as sitk

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
def collate_function(data):
    """
    	:data: a list for a batch of samples. [[tensor,string,], ..., [tensor,string]]
    	"""
    transposed_data = list(zip(*data))
    max_min,path,ylab,xlab,y, x  =  transposed_data[5],transposed_data[4], transposed_data[3],transposed_data[2],transposed_data[1], transposed_data[0]
    x = torch.stack(x, 0)
    y = torch.stack(y, 0)
    xlab = torch.stack(xlab, 0)
    ylab = torch.stack(ylab, 0)
    # labels = torch.stack(labels, 0)
    return (x,y,xlab,ylab,path,max_min)

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
    model.load_state_dict(torch.load(r"D:\mx\CT_Re\GroupMorph\experiments\GroupMorph_Ncccheckpoint.pt"))

    transfor = SpatialTransformer()#.cuda()


    valdataset = getdirfile('test')
    valdata = Ourdata_test(valdataset,None,None)

    valdata = Data.DataLoader(valdata, batch_size=2, shuffle=False,collate_fn=collate_function)


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
            # smo_loss = smo
            X_Y_label = transfor(X_label, flows, mode='nearest')

            warp_img = warps.cpu().detach().numpy()
            def_out = X_Y_label.cpu().detach().numpy()
            filename = data[4]
            max_min = data[5]

            for index in range(warps.shape[0]):
                img = warp_img[index, 0]
                mm = max_min[index]
                if mm[1] == 0:
                    img = (img * mm[0]).astype(np.int16)
                else:
                    img = (img * (mm[0] - mm[1]) + mm[1]).astype(np.int16)
                seg = def_out[index]
                seg[seg < 0.5] = 0
                seg[seg > 0] = 1
                s = np.zeros_like(img)
                s[seg[0] == 1] = 1
                s[seg[1] == 1] = 2
                s[seg[2] == 1] = 3
                s[seg[3] == 1] = 4

                file = filename[index]
                file = file.split(os.path.sep)
                file.insert(-2, 'GroupmorphNcc')
                file[-4] = 'test'
                if not os.path.exists(os.path.sep.join(file[:-1])):
                    os.makedirs(os.path.sep.join(file[:-1]))
                img = sitk.GetImageFromArray(img)
                s = sitk.GetImageFromArray(s)
                sitk.WriteImage(img, os.path.sep.join(file))
                file[-1] = 'Lab' + file[-1][2:]
                sitk.WriteImage(s, os.path.sep.join(file))



if __name__ == '__main__':
    start = datetime.datetime.now()
    train()
    end = datetime.datetime.now()
    print("Time used:", end - start)
