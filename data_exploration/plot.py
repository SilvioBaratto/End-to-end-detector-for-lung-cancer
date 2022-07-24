import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
from dsets_segmentation import getCandidateInfoList, getCt
from model_segmentation import SegmentationMask, MaskTuple
from vis import build2dLungMask
from util.util import xyz2irc
import scipy.ndimage.morphology

candidateInfo_list = getCandidateInfoList(requireOnDisk_bool=False)
candidateInfo_list[0]

clim=(0, 1.3)
start_ndx = 3
# mask_model = SegmentationMask().to('cuda')
mask_model = SegmentationMask()

nit_ndx = 0
candidateInfo_tup = candidateInfo_list[nit_ndx]

ct = getCt(candidateInfo_tup.series_uid)
center_irc = xyz2irc(candidateInfo_tup.center_xyz, ct.origin_xyz, ct.vxSize_xyz, ct.direction_a)
print(candidateInfo_tup, 'center_irc', center_irc)

mask_tup = build2dLungMask(ct.series_uid, int(center_irc.index))
mask_tup = mask_tup._make(x.cpu().numpy()[0][0] for x in mask_tup)

fig = plt.figure(figsize=(40,10))

subplot = fig.add_subplot(1, 4, 1)
subplot.set_title('lung mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.lung_mask, clim=(0,1), cmap=tgray)
plt.imsave('test_1.png', ct.hu_a[int(center_irc.index)], cmap = 'gray')

subplot = fig.add_subplot(1, 4, 2)
subplot.set_title('candidate mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.candidate_mask, clim=(0,1), cmap=tgray)
plt.imsave('test_2.png', ct.hu_a[int(center_irc.index)], cmap = 'gray')

subplot = fig.add_subplot(1, 4, 3)
subplot.set_title('ben mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.neg_mask, clim=(0,1), cmap=tgray)
plt.imsave('test_3.png', ct.hu_a[int(center_irc.index)], cmap = 'gray')

subplot = fig.add_subplot(1, 4, 4)
subplot.set_title('mal mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.pos_mask, clim=(0,1), cmap=tgray)
plt.imsave('test_3.png', ct.hu_a[int(center_irc.index)], cmap = 'gray')