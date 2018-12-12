import os
import numpy as np
import chainer
from chainercv.links import SSD300

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
label_names = ['cup']

pretrained_model = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model_epoch_2000')
model = SSD300(n_fg_class=1, pretrained_model=pretrained_model)

def handler(_itr, ctx):
    for img in _itr:
        img = img.transpose(2, 0, 1)
        bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        result = []
        for b, lbl, s in zip(bbox, label, score):
            r = {'box': b.tolist(),
                 'label': label_names[lbl],
                 'score': float(s)}
            result.append(r)
        yield result
