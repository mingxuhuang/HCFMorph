import os, glob
import torch, sys
from matplotlib.pyplot import ylabel
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import SimpleITK as sitk

import numpy as np
# sys.path.append(r"D:\mx\CT_Re")

def resize_image(itkimage, newSize, resamplemethod=sitk.sitkBSpline):
    # print(itkimage.GetSize())
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()

    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)    # spacing肯定不能是整数

    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_res = resampler.Execute(itkimage)  # 得到重新采样后的图像
    # itk_img_res.SetSpacing((1,1,3))
    # print(itk_img_res.GetSize())
    return itk_img_res

class Ourdata(Dataset):
    def __init__(self, data_path, transforms,resize = (128,128,32)):
        self.paths = data_path
        self.transforms = transforms
        self.resize = resize


    def __getitem__(self, index):
        path = self.paths[index]
        CT1 = sitk.ReadImage(path[1])
        CT2 = sitk.ReadImage(path[0])
        if self.resize is not None:
            CT1 = resize_image(CT1, self.resize)
            CT2 = resize_image(CT2, self.resize)
        x = sitk.GetArrayFromImage(CT1)
        y = sitk.GetArrayFromImage(CT2)
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        CT1labelpath = path[1].split(os.path.sep)
        CT1labelpath[-1] = 'Lab'+CT1labelpath[-1][2:]
        CT2labelpath = path[0].split(os.path.sep)
        CT2labelpath[-1] = 'Lab'+CT2labelpath[-1][2:]
        CT1lab = sitk.ReadImage(os.path.sep.join(CT1labelpath))
        CT2lab = sitk.ReadImage(os.path.sep.join(CT2labelpath))
        CTlab = sitk.GetArrayFromImage(CT1lab)
        MRlab = sitk.GetArrayFromImage(CT2lab)
        xlab = np.zeros_like(CTlab)
        xlab = np.expand_dims(xlab, axis=0).repeat(4, axis=0)
        ylab = np.zeros_like(xlab)
        xlab[0] = (CTlab == 1).astype(np.uint8)
        xlab[1] = (CTlab == 2).astype(np.uint8)
        xlab[2] = (CTlab == 3).astype(np.uint8)
        xlab[3] = (CTlab == 4).astype(np.uint8)

        ylab[0] = (MRlab == 1).astype(np.uint8)
        ylab[1] = (MRlab == 2).astype(np.uint8)
        ylab[2] = (MRlab == 3).astype(np.uint8)
        ylab[3] = (MRlab == 4).astype(np.uint8)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        # xlab = np.expand_dims(xlab, axis=0)
        # ylab = np.expand_dims(ylab, axis=0)
        return x, y,xlab.astype(np.float32),ylab.astype(np.float32)

    def __len__(self):
        return len(self.paths)


class ACDCdata(Dataset):
    def __init__(self, data_path, transforms,resize = (128,128,32)):
        self.paths = data_path
        self.transforms = transforms
        self.resize = resize


    def __getitem__(self, index):
        path = self.paths[index]
        CT1 = sitk.ReadImage(os.path.join(r'D:\mx\CT_Re',path[1]))
        CT2 = sitk.ReadImage(os.path.join(r'D:\mx\CT_Re',path[0]))
        if self.resize is not None:
            CT1 = resize_image(CT1, self.resize)
            CT2 = resize_image(CT2, self.resize)
        x = sitk.GetArrayFromImage(CT1)
        y = sitk.GetArrayFromImage(CT2)
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        CT1labelpath = path[1].replace("img", "lab")
        CT2labelpath = path[0].replace("img", "lab")
        CT1lab = sitk.ReadImage(os.path.join(r'D:\mx\CT_Re',CT1labelpath))
        CT2lab = sitk.ReadImage(os.path.join(r'D:\mx\CT_Re',CT2labelpath))
        CTlab = sitk.GetArrayFromImage(CT1lab)
        MRlab = sitk.GetArrayFromImage(CT2lab)
        xlab = np.zeros_like(CTlab)
        xlab = np.expand_dims(xlab, axis=0).repeat(3, axis=0)
        ylab = np.zeros_like(xlab)
        xlab[0] = (CTlab == 1).astype(np.uint8)
        xlab[1] = (CTlab == 2).astype(np.uint8)
        xlab[2] = (CTlab == 3).astype(np.uint8)

        ylab[0] = (MRlab == 1).astype(np.uint8)
        ylab[1] = (MRlab == 2).astype(np.uint8)
        ylab[2] = (MRlab == 3).astype(np.uint8)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        # xlab = np.expand_dims(xlab, axis=0)
        # ylab = np.expand_dims(ylab, axis=0)
        return x, y,xlab.astype(np.float32),ylab.astype(np.float32)

    def __len__(self):
        return len(self.paths)


class ABDdata(Dataset):
    def __init__(self, data_path, transforms,resize = (128,128,32)):
        self.paths = data_path
        self.transforms = transforms
        self.resize = resize


    def __getitem__(self, index):
        path = self.paths[index]
        CT1 = sitk.ReadImage(path[1])
        CT2 = sitk.ReadImage(path[0])
        if self.resize is not None:
            CT1 = resize_image(CT1, self.resize)
            CT2 = resize_image(CT2, self.resize)
        x = sitk.GetArrayFromImage(CT1)
        y = sitk.GetArrayFromImage(CT2)
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        CT1labelpath = path[1].replace("img", "lab")
        CT2labelpath = path[0].replace("img", "lab")
        CT1lab = sitk.ReadImage(CT1labelpath)
        CT2lab = sitk.ReadImage(CT2labelpath)
        CTlab = sitk.GetArrayFromImage(CT1lab)
        MRlab = sitk.GetArrayFromImage(CT2lab)
        xlab = np.zeros_like(CTlab)
        xlab = np.expand_dims(xlab, axis=0).repeat(4, axis=0)
        ylab = np.zeros_like(xlab)
        xlab[0] = (CTlab == 1).astype(np.uint8)
        xlab[1] = (CTlab == 2).astype(np.uint8)
        xlab[2] = (CTlab == 3).astype(np.uint8)
        xlab[3] = (CTlab == 4).astype(np.uint8)

        ylab[0] = (MRlab == 1).astype(np.uint8)
        ylab[1] = (MRlab == 2).astype(np.uint8)
        ylab[2] = (MRlab == 3).astype(np.uint8)
        ylab[3] = (MRlab == 4).astype(np.uint8)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        # xlab = np.expand_dims(xlab, axis=0)
        # ylab = np.expand_dims(ylab, axis=0)
        return x, y,xlab.astype(np.float32),ylab.astype(np.float32)

    def __len__(self):
        return len(self.paths)



class IXIdata(Dataset):
    def __init__(self, data_path, transforms, resize=(128, 128, 128)):
        self.paths = data_path
        self.transforms = transforms
        self.resize = resize

    def __getitem__(self, index):
        # print(os.getcwd())
        path = self.paths[index]
        CT1 = sitk.ReadImage(path[1])
        CT2 = sitk.ReadImage(path[0])
        if self.resize is not None:
            CT1 = resize_image(CT1, self.resize)
            CT2 = resize_image(CT2, self.resize)
        x = sitk.GetArrayFromImage(CT1)
        y = sitk.GetArrayFromImage(CT2)
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        CT1labelpath = path[1].replace("img", "lab")
        CT2labelpath = path[0].replace("img", "lab")
        CT1lab = sitk.ReadImage(CT1labelpath)
        CT2lab = sitk.ReadImage(CT2labelpath)
        CTlab = sitk.GetArrayFromImage(CT1lab)
        MRlab = sitk.GetArrayFromImage(CT2lab)
        xlab = np.zeros_like(CTlab)
        xlab = np.expand_dims(xlab, axis=0).repeat(4, axis=0)
        ylab = np.zeros_like(xlab)
        xlab[0] = (CTlab == 1).astype(np.uint8)
        xlab[1] = (CTlab == 2).astype(np.uint8)
        xlab[2] = (CTlab == 3).astype(np.uint8)
        xlab[3] = (CTlab == 4).astype(np.uint8)

        ylab[0] = (MRlab == 1).astype(np.uint8)
        ylab[1] = (MRlab == 2).astype(np.uint8)
        ylab[2] = (MRlab == 3).astype(np.uint8)
        ylab[3] = (MRlab == 4).astype(np.uint8)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        # xlab = np.expand_dims(xlab, axis=0)
        # ylab = np.expand_dims(ylab, axis=0)
        return x, y, xlab.astype(np.float32), ylab.astype(np.float32)

    def __len__(self):
        return len(self.paths)


class Ourdata_test(Dataset):
    def __init__(self, data_path, transforms,resize = (128,128,32)):
        self.paths = data_path
        self.transforms = transforms
        self.resize = resize


    def __getitem__(self, index):
        path = self.paths[index]
        CT1 = sitk.ReadImage(path[1])
        CT2 = sitk.ReadImage(path[0])
        if self.resize is not None:
            CT1 = resize_image(CT1, self.resize)
            CT2 = resize_image(CT2, self.resize)
        x = sitk.GetArrayFromImage(CT1)
        y = sitk.GetArrayFromImage(CT2)
        max_min = [x.max(), x.min()]
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        CT1labelpath = path[1].split(os.path.sep)
        CT1labelpath[-1] = 'Lab'+CT1labelpath[-1][2:]
        CT2labelpath = path[0].split(os.path.sep)
        CT2labelpath[-1] = 'Lab'+CT2labelpath[-1][2:]
        CT1lab = sitk.ReadImage(os.path.sep.join(CT1labelpath))
        CT2lab = sitk.ReadImage(os.path.sep.join(CT2labelpath))
        CTlab = sitk.GetArrayFromImage(CT1lab)
        MRlab = sitk.GetArrayFromImage(CT2lab)
        xlab = np.zeros_like(CTlab)
        xlab = np.expand_dims(xlab, axis=0).repeat(4, axis=0)
        ylab = np.zeros_like(xlab)
        xlab[0] = (CTlab == 1).astype(np.uint8)
        xlab[1] = (CTlab == 2).astype(np.uint8)
        xlab[2] = (CTlab == 3).astype(np.uint8)
        xlab[3] = (CTlab == 4).astype(np.uint8)

        ylab[0] = (MRlab == 1).astype(np.uint8)
        ylab[1] = (MRlab == 2).astype(np.uint8)
        ylab[2] = (MRlab == 3).astype(np.uint8)
        ylab[3] = (MRlab == 4).astype(np.uint8)


        x,y = torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)
        xlab = torch.from_numpy(xlab).float()
        ylab = torch.from_numpy(ylab).float()

        return x, y,xlab,ylab,path[1],max_min

    def __len__(self):
        return len(self.paths)

class ACDCdata_test(Dataset):
    def __init__(self, data_path, transforms,resize = (128,128,32)):
        self.paths = data_path
        self.transforms = transforms
        self.resize = resize


    def __getitem__(self, index):
        path = self.paths[index]
        CT1 = sitk.ReadImage(os.path.join(r'D:\mx\CT_Re',path[1]))
        CT2 = sitk.ReadImage(os.path.join(r'D:\mx\CT_Re',path[0]))
        if self.resize is not None:
            CT1 = resize_image(CT1, self.resize)
            CT2 = resize_image(CT2, self.resize)
        x = sitk.GetArrayFromImage(CT1)
        y = sitk.GetArrayFromImage(CT2)
        max_min = [x.max(), x.min()]
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        CT1labelpath = path[1].replace("img", "lab")
        CT2labelpath = path[0].replace("img", "lab")
        CT1lab = sitk.ReadImage(os.path.join(r'D:\mx\CT_Re',CT1labelpath))
        CT2lab = sitk.ReadImage(os.path.join(r'D:\mx\CT_Re',CT2labelpath))
        CTlab = sitk.GetArrayFromImage(CT1lab)
        MRlab = sitk.GetArrayFromImage(CT2lab)
        xlab = np.zeros_like(CTlab)
        xlab = np.expand_dims(xlab, axis=0).repeat(3, axis=0)
        ylab = np.zeros_like(xlab)
        xlab[0] = (CTlab == 1).astype(np.uint8)
        xlab[1] = (CTlab == 2).astype(np.uint8)
        xlab[2] = (CTlab == 3).astype(np.uint8)

        ylab[0] = (MRlab == 1).astype(np.uint8)
        ylab[1] = (MRlab == 2).astype(np.uint8)
        ylab[2] = (MRlab == 3).astype(np.uint8)


        x,y = torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)
        xlab = torch.from_numpy(xlab).float()
        ylab = torch.from_numpy(ylab).float()

        return x, y,xlab,ylab,os.path.join(r'D:\mx\CT_Re',path[1]),max_min

    def __len__(self):
        return len(self.paths)