"""Training code of Single Shot Multibox Detector
Original author: Preferred Networks, Inc.
https://github.com/chainer/chainercv/blob/master/examples/ssd/train.py
Updated by: ABEJA, Inc.
"""

import os
import io
import copy
import numpy as np

from PIL import Image

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv import transforms
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

from abeja.datasets import Client

from utils.callbacks import Statistics
from utils.tensorboard import Tensorboard
from dataset import load_dataset_from_api
from dataset import DetectionDatasetFromAPI

nb_epochs = 2000

BATCHSIZE = int(os.environ.get('BATCHSIZE', '8'))
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
USE_GPU = int(os.environ.get('USE_GPU', '-1'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label

def copy_ssd(model, premodel):
    _ = premodel(np.zeros((1, 3, 300, 300), np.float32))
    _ = model(np.zeros((1, 3, 300, 300), np.float32))

    extractor_src = premodel.__dict__['extractor']
    multibox_src = premodel.__dict__['multibox']
    extractor_dst = model.__dict__['extractor']
    multibox_dst = model.__dict__['multibox']

    layers = extractor_src.__dict__['_children']
    for l in layers:
        if l == 'norm4':
            if extractor_dst[l].scale.shape == extractor_src[l].scale.shape:
                extractor_dst[l].copyparams(extractor_src[l])
        else:
            if extractor_dst[l].W.shape == extractor_src[l].W.shape:
                extractor_dst[l].copyparams(extractor_src[l])

    for c_src, c_dst in zip(multibox_src['conf'], multibox_dst['conf']):
        if c_dst.W.data.shape == c_src.W.data.shape:
            c_dst.copyparams(c_src)

    for c_src, c_dst in zip(multibox_src['loc'], multibox_dst['loc']):
        if c_dst.W.data.shape == c_src.W.data.shape:
            c_dst.copyparams(c_src)

def fix_ssd(train_chain):
    names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
        'conv3_1', 'conv3_2', 'conv3_3',
        'conv4_1', 'conv4_2', 'conv4_3',
        'conv5_1', 'conv5_2', 'conv5_3',
        'conv6', 'conv7',
        'norm4']
    d = train_chain.model.extractor.__dict__
    for name in train_chain.model.extractor._children:
        if name in names:
            layer = d[name]
            layer.disable_update()

def handler(context):
    dataset_alias = context.datasets
    data = list(load_dataset_from_api(dataset_alias['train']))
    
    np.random.seed(0)
    data = np.random.permutation(data)
    nb_data = len(data)
    nb_train = int(7 * nb_data // 10)
    train_data_raw = data[:nb_train]
    test_data_raw = data[nb_train:]

    premodel = SSD300(n_fg_class=20, pretrained_model='voc0712')
    model = SSD300(n_fg_class=1)

    copy_ssd(model, premodel)

    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if USE_GPU >= 0:
        chainer.cuda.get_device_from_id(USE_GPU).use()
        model.to_gpu()

     # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    fix_ssd(train_chain)
    
    train_data = DetectionDatasetFromAPI(train_data_raw)
    test_data = DetectionDatasetFromAPI(test_data_raw, use_difficult=True, return_difficult=True)

    train_data = TransformDataset(train_data, Transform(model.coder, model.insize, model.mean))
    train_iter = chainer.iterators.SerialIterator(train_data, BATCHSIZE)

    test_iter = chainer.iterators.SerialIterator(
            test_data, BATCHSIZE, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=USE_GPU)
    trainer = training.Trainer(updater, (nb_epochs, 'epoch'), out=ABEJA_TRAINING_RESULT_DIR)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([1200, 1600], 'epoch'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=['cup']),
        trigger=(1, 'epoch'))

    log_interval = 1, 'epoch'
    trainer.extend(extensions.LogReport(trigger=log_interval))

    print_entries = ['epoch', 
                     'main/loss', 'main/loss/loc', 'main/loss/conf',
                     'validation/main/map']
    report_entries = ['epoch', 'lr',
                      'main/loss', 'main/loss/loc', 'main/loss/conf',
                      'validation/main/map']
    
    trainer.extend(Statistics(report_entries, nb_epochs), trigger=log_interval)
    trainer.extend(Tensorboard(report_entries, out_dir=log_path))
    trainer.extend(extensions.PrintReport(print_entries), trigger=log_interval)

    trainer.extend(
        extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'),
        trigger=(nb_epochs, 'epoch'))

    trainer.run()
