import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append("..")

from dataset.dsets_segmentation import getCandidateInfoList, getCt
from model.model_segmentation import SegmentationMask, MaskTuple
from analysis.vis import build2dLungMask
from util.util import xyz2irc

candidateInfo_list = getCandidateInfoList(requireOnDisk_bool=False)
candidateInfo_list[0]
series_list = sorted(set(t.series_uid for t in candidateInfo_list))

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = copy.deepcopy(cmap)
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.75, N+4)
    return mycmap

tgray = transparent_cmap(plt.cm.gray)
tpurp = transparent_cmap(plt.cm.Purples)
tblue = transparent_cmap(plt.cm.Blues)
tgreen = transparent_cmap(plt.cm.Greens)
torange = transparent_cmap(plt.cm.Oranges)
tred = transparent_cmap(plt.cm.Reds)


clim=(0, 1.3)
start_ndx = 3
mask_model = SegmentationMask().to('cuda')

ct_list = []
for nit_ndx in range(start_ndx, start_ndx+3):
    candidateInfo_tup = candidateInfo_list[nit_ndx]
    ct = getCt(candidateInfo_tup.series_uid)
    center_irc = xyz2irc(candidateInfo_tup.center_xyz, ct.origin_xyz, ct.vxSize_xyz, ct.direction_a)
    
    ct_list.append((ct, center_irc))
start_ndx = nit_ndx + 1

fig = plt.figure(figsize=(60,90))
subplot_ndx = 0 
for ct_ndx, (ct, center_irc) in enumerate(ct_list):
    mask_tup = build2dLungMask(ct.series_uid, int(center_irc.index))
    
#    ct_g = torch.from_numpy(ct.hu_a[int(center_irc.index)].astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda')
#    pos_g = torch.from_numpy(ct.positive_mask[int(center_irc.index)].astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda')
#    input_g = ct_g / 1000
    
#    label_g, neg_g, pos_g, lung_mask, mask_dict = mask_model(input_g, pos_g)
#    mask_tup = MaskTuple(**mask_dict)
    for attr_ndx, attr_str in enumerate(mask_tup._fields):

        subplot_ndx = 1 + 3 * 2 * attr_ndx + 2 * ct_ndx
        subplot = fig.add_subplot(len(mask_tup), len(ct_list)*2, subplot_ndx)
        subplot.set_title(attr_str)
        
        
        #print(layer_func, ct.hu_a.shape, layer_func(ct, mask_tup, int(center_irc.index)).shape, center_irc.index)

        # plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 3000), cmap='RdGy')
        # plt.imshow(mask_tup[attr_ndx][0][0].cpu(), clim=clim, cmap=tblue)

        plt.imsave('test.png', ct.hu_a[int(center_irc.index)], cmap = 'RdGy')
        
        #if attr_ndx == 1: break
    #break

