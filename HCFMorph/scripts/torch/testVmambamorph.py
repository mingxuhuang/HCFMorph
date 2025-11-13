#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019.

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu.
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os

import random
import argparse
import time
import warnings
import matplotlib
from monai.networks.blocks.backbone_fpn_utils import torchvision_models
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import math
import datetime
from torch.utils.tensorboard import SummaryWriter
from contextlib import contextmanager
from GroupMorph.Model.Net import GruopMorph
from HCFMorph.mambamorph.torch.RDP import RDP
from HCFMorph.mambamorph.torch.Corr import CorrMLP
# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8, from packages instead of source code
from voxelmorph.torch.layers import SpatialTransformer
import sys
import pickle
import SimpleITK as sitk
sys.path.append(r"D:\mx\CT_Re\VMambaMorph\mambamorph")
import voxelmorph.torch.layers as layers
from HCFMorph.mambamorph.torch.vxmorph import VoxelMorph

fullsize = layers.ResizeTransform(1 / 2, 3)



# import torch
# data = torch.randn([4,32,192,320,1])
# out = src_generators.chunk_volume(data)
# a = 0


sys.path.append(r"D:\mx\CT_Re\VMambaMorph\mambamorph/torch")

import HCFMorph.mambamorph.torch.losses as src_loss
import HCFMorph.mambamorph.torch.networks as networks
import HCFMorph.mambamorph.torch.utils as utils
from HCFMorph.mambamorph.torch.TransMorph import CONFIGS as CONFIGS_TM
import HCFMorph.mambamorph.torch.TransMorph as TransMorph


from getourfile import getdirfile
from getdata import Ourdata_test,ACDCdata_test
import json

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--subj-train', default="/media/ziyang/14TBWD/VMambaMorph/SR-Reg_v2/train_subj_v2.pkl",
                    help='subjects used for train')
parser.add_argument('--subj-val', default="/media/ziyang/14TBWD/VMambaMorph/SR-Reg_v2/val_subj_v2.pkl",
                    help='subjects used for validation')
parser.add_argument('--subj-test', default="/media/ziyang/14TBWD/VMambaMorph/SR-Reg_v2/test_subj_v2.pkl",
                    help='subjects used for test')
parser.add_argument('--vol-path', default="/media/ziyang/14TBWD/VMambaMorph/SR-Reg_v2/vol/",
                    help='path to cross modality volume')
parser.add_argument('--seg-path', default="/media/ziyang/14TBWD/VMambaMorph/SR-Reg_v2/seg/",
                    help='path to cross modality segmentation')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--mode', default='mr>ct', help='register from mr -> ct')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--scale', type=float, default=1.0, help='scale factor of the original volume')
parser.add_argument('--chunk', action='store_true', help='whether to use chunk the volumes')

# training parameters
parser.add_argument('--gpu', default=None, help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=2, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--load-model-dds', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--warm-up', type=float, default=0, help='rate of warm up epochs')
parser.add_argument('--no-amp', action='store_true', help='NOT auto mix precision training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--model', type=str, default='vimm', help='Choose a model to train (vm, vm-feat, mm, mm-feat)')

# loss hyperparameters
parser.add_argument('--image-loss', default='dice',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.1,
                    help='weight of deformation loss (default: 0.1)')
parser.add_argument('--ignore-label', type=int, nargs='+', default=[0, 5, 24],
                    help='list of ignorable labels')
parser.add_argument('--cl', type=float, default=0.0, help='whether to use contrastive loss and set its weight')
args = parser.parse_args()
bidir = args.bidir
args.epochs = 600
@contextmanager
def conditional_autocast(enabled: bool):
    if enabled:
        with torch.amp.autocast('cuda'):
            yield
    else:
        yield


# load and prepare data
# with open(args.subj_train, 'rb') as file:
#     train_subject = pickle.load(file)
# with open(args.subj_val, 'rb') as file:
#     val_subject = pickle.load(file)
# with open(args.subj_test, 'rb') as file:
#     test_subject = pickle.load(file)
#
# data_example = vxm.py.utils.load_volfile(
#     args.seg_path + train_subject[0] + '.nii.gz')
# inshape = tuple([int(old_size * args.scale) for old_size in data_example.shape])
# labels_in = np.unique(data_example)
#
if args.chunk:
    assert args.scale == 0.5, "If using chunking opearation, the scale must be 0.5!"
    args.scale *= 2

add_feat_axis = not args.multichannel

# prepare model folder
model_dir = r'D:\mx\CT_Re\VMambaMorph\models'#args.model_dir
if os.path.exists(model_dir):
    warnings.warn("Ensure that you don't overwrite the former folder!")
# assert not os.path.exists(model_dir), "Ensure that you don't overwrite the former folder!"
os.makedirs(model_dir, exist_ok=True)
model_list= ['RDP','Corr','vm-feat','MCMO-feat','tm-feat','mm-feat','vimm-feat','mmC-feat','MCMOfusion-feat','GM']
datasetname = 'ACDC'#'Our'
if datasetname=='ACDC':
    with open(r"D:\mx\CT_Re\ACDCdata.json", 'r') as f:  #train_5
        data = json.load(f)
    # train_subject = data['train']
    val_subject = data['test']#getdirfile('test')
    inshape = (32,192,192)#(32,192,320)
else:
    val_subject = getdirfile('test')
    inshape =  (32,192,320)

for modelname in model_list[-2:-1]:
    flowname = 'preint_flow' if modelname in ['tm-feat','GM'] else  'pos_flow'  #flowname = ['preint_flow','pos_flow'],tm,gm没posflows

    # args.model = 'MCMO-feat'
    # args.model = 'vm-feat'
    # args.model = 'tm-feat'
    # args.model = 'mm-feat'
    # args.model = 'rvm'
    # args.model = 'vimm-feat'
    # args.model = 'mmC-feat'
    # args.model = 'MC-feat'



    args.model = modelname
    # args.model = 'GM'
    name = args.model
    args.mode = 'ct>ct'

    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num),args.model)
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    device = torch.device('cuda:' + str(GPU_iden) if GPU_avai else 'cpu')
    # unet architecture
    enc_nf = args.enc if args.enc else [16] * 4
    dec_nf = args.dec if args.dec else [16] * 6


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

    # Define a model
    if args.model == 'vm':
        model = networks.VxmDense.load(args.load_model, device) \
            if args.load_model else \
            networks.VxmDense(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=bidir,
                int_steps=args.int_steps,
                int_downsize=args.int_downsize,
            )
    elif args.model == 'vm-feat':
        model = networks.VxmFeat.load(args.load_model, device) \
            if args.load_model else \
            networks.VxmFeat(
                inshape=inshape,
                nb_feat_extractor=[[16] * 2, [16] * 4],
                nb_unet_features=[enc_nf, dec_nf],
                bidir=bidir,
                int_steps=args.int_steps,
                int_downsize=args.int_downsize,
            )
        paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\\'+datasetname+'\modelsvm-feat\min_val.pt')


    elif args.model == 'tm':
        config = CONFIGS_TM['TransMorph']
        config.img_size = inshape
        model = TransMorph.TransMorph(config)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))

    elif args.model == 'tm-feat':
        config = CONFIGS_TM['TransMorph']
        config.img_size = inshape
        model = TransMorph.TransMorphFeat(config)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
        paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\\'+datasetname+'\modelstm-feat\min_val.pt')

    elif args.model == 'mm':
        config = CONFIGS_TM['MambaMorph']
        config.img_size = inshape
        model = TransMorph.MambaMorph(config)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
    elif args.model == 'mm-feat':
        config = CONFIGS_TM['MambaMorph']
        config.img_size = inshape
        model = TransMorph.MambaMorphFeat(config)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
        paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\\'+datasetname+'\modelsmm-feat\min_val.pt')

    elif args.model == 'vimm':
        config = CONFIGS_TM['VMambaMorph']
        config.img_size = inshape
        model = TransMorph.VMambaMorph(config)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
    elif args.model == 'vimm-feat':
        config = CONFIGS_TM['VMambaMorph']
        config.img_size = inshape
        model = TransMorph.VMambaMorphFeat(config)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
        paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\\'+datasetname+'\modelsvimm-feat\min_val.pt')

    elif args.model == 'rvm':
        config = CONFIGS_TM['VMambaMorph']
        config.img_size = inshape
        model = TransMorph.RecVMambaMorphFeat(config)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
        paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\models\min_val.pt')

    elif args.model == 'HCFMorph':
        config = CONFIGS_TM['MambaMorph']
        config.img_size = inshape
        config.base_resolution = [i//4 for i in inshape]
        model = TransMorph.MambaCMoFusionMorphFeat(config)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))

        # paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\\'+datasetname+'\modelsMCMOF-feat\min_val.pt')
        # paras = torch.load(r"D:\mx\CT_Re\VMambaMorph\modelsMCMOF-feat_NonLocal\min_val.pt")
        paras = torch.load(r"D:\mx\CT_Re\VMambaMorph\modelsMCMOF-feat_NonLocal+FFE_ACDC\min_val.pt")
    elif args.model == 'GM':
        GMimgshape = inshape # (160, 192, 192)
        GMgroups = (4, 2, 2)  # (4,4,4), (4,4,2), (4,2,2) or (2,2,2)
        model = GruopMorph(1, 8, GMimgshape, GMgroups)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
        paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\\'+datasetname+'\modelsGM\min_val.pt')
    elif args.model == 'RDP':
        imgshape = inshape#(32,160,160)#(32, 192, 320)
        model = RDP(inshape)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
        paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\\' + datasetname + '\modelsRDP\min_val.pt')


    elif args.model == 'Corr':
        imgshape = inshape#(32,160,160)#(32, 192, 320)
        model = CorrMLP()
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
        paras = torch.load(r'D:\mx\CT_Re\VMambaMorph\\'+datasetname+'\modelsCorr\min_val.pt')




    # tem = model.state_dict()

    model.load_state_dict(paras,strict=False)
    model.to(device)
    transform_model = SpatialTransformer(inshape, mode='bilinear')  # STN #'bilinear'
    transform_model.to(device)

    if datasetname == 'ACDC':
        valset =ACDCdata_test(val_subject, None,None)#Ourdata_test(val_subject, None,None)
    else:
        valset = Ourdata_test(val_subject, None,None)
    valdata = DataLoader(valset, batch_size=args.batch_size, shuffle=False,collate_fn=collate_function)

    if not args.no_amp:
        scaler = torch.amp.GradScaler('cuda')


    with torch.no_grad():
        model.eval()

        for val_step, vdata in enumerate(valdata):
            inputs = [vdata[0], vdata[1]]  # src_img, tgt_img
            src_label = vdata[2]

            src_label = src_label.to(device).float()
            inputs = [d.to(device).float() for d in inputs]  # volume pairs

            # run inputs through the model to produce a warped image and flow field
            with conditional_autocast(not args.no_amp):
                ret_dict = model(*inputs, return_pos_flow=True)
                warped_vol = ret_dict['moved_vol']
                preint_flow = ret_dict[flowname]
                if type(warped_vol) == list:
                    warped_vol = warped_vol[0]
                    preint_flow = preint_flow[0]
                if preint_flow.size(2) !=32:
                    preint_flow = fullsize(preint_flow)

                warped_label = transform_model(src_label, preint_flow)
                y_pred = (warped_label, preint_flow)
                warped_vol = transform_model(inputs[0], preint_flow)
                warp_img = warped_vol.cpu().detach().numpy()
                def_out = warped_label.cpu().detach().numpy()
                preint_flow = preint_flow.cpu().detach().numpy()
                filename = vdata[4]
                max_min = vdata[5]

                for index in range(warped_vol.shape[0]):
                    flow = preint_flow[index].astype(np.float32)
                    img = warp_img[index,0]
                    mm = max_min[index]
                    if mm[1]==0:
                        img = (img*mm[0]).astype(np.int16)
                    else:
                        img = (img*(mm[0]-mm[1])+mm[1]).astype(np.int16)
                    seg = def_out[index]
                    seg[seg<0.5] = 0
                    seg[seg>0] = 1
                    s = np.zeros_like(img)
                    s[seg[0]==1] = 1
                    s[seg[1]==1] = 2
                    s[seg[2]==1] = 3


                    file = os.path.normpath(filename[index])
                    #CTRE
                    if datasetname!='ACDC':
                        s[seg[3]==1] = 4 #ACDC 没有4
                        file = file.split(os.path.sep)
                        file.insert(-2,name)
                        file[-4] = 'test'
                        if not os.path.exists(os.path.sep.join(file[:-1])):
                            os.makedirs(os.path.sep.join(file[:-1]))

                        img = sitk.GetImageFromArray(img)
                        s = sitk.GetImageFromArray(s)
                        sitk.WriteImage(img, os.path.sep.join(file))  # CTRE
                        file[-1] = 'Lab' + file[-1][2:]  # CTRE
                        sitk.WriteImage(s,os.path.sep.join(file))#CTRE
                        fl = sitk.GetImageFromArray(flow)
                        file[-1] = file[-1].replace('Lab', 'flow')
                        sitk.WriteImage(fl, os.path.sep.join(file))


                    #ACDC
                    else:
                        file = file.replace('\\data','\\test\\'+name)
                        os.makedirs(os.path.dirname(file), exist_ok=True)
                        img = sitk.GetImageFromArray(img)
                        s = sitk.GetImageFromArray(s)
                        fl = sitk.GetImageFromArray(flow)
                        sitk.WriteImage(img, file)
                        file = file.replace('img','lab')
                        sitk.WriteImage(s, file)
                        file = file.replace('lab','flow')
                        sitk.WriteImage(fl, file)
print('Done')


