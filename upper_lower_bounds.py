# -*- coding: utf-8 -*-

import json
import numpy as np
import random
from scipy.stats import pearsonr
from evaluation import *
import numpy as np

labels = json.load(open("./data/labels.json", 'r'))

gt = {}
for key, val in labels.items():
    pi = [v for v in val['pinch'].values()]
    c = [v for v in val['clench'].values()]
    po = [v for v in val['poke'].values()]
    pa = [v for v in val['palm'].values()]
    f_ = [v for v in val['familiarity'].values()]
    f = []
    for e in f_:
        f += e

    piv = np.average(pi)
    cv = np.average(c)
    pov = np.average(po)
    pav = np.average(pa)
    fv = np.average(f)
    val = {'pinch': piv, 'clench': cv, 'poke': pov, 'palm': pav, 'familiarity': fv}
    gt[key] = val

mse_list = []
corr_list = []
acc_list = []
for _ in range(1000):
    out = {}
    for key, val in labels.items():
        pi = random.randint(0, 100)
        c = random.randint(0, 100)
        po = random.randint(0, 100)
        pa = random.randint(0, 100)
        f = random.randint(0, 100)

        val = {'pinch': pi, 'clench': c, 'poke': po, 'palm': pa, 'familiarity': f}
        out[key] = val

    pinch_gt = []
    pinch_pred = []
    clench_gt = []
    clench_pred = []
    poke_gt = []
    poke_pred = []
    palm_gt = []
    palm_pred = []
    for k in out.keys():
        pinch_gt.append(gt[k]['pinch'])
        pinch_pred.append(out[k]['pinch'])
        clench_gt.append(gt[k]['clench'])
        clench_pred.append(out[k]['clench'])
        poke_gt.append(gt[k]['poke'])
        poke_pred.append(out[k]['poke'])
        palm_gt.append(gt[k]['palm'])
        palm_pred.append(out[k]['palm'])

    gt_array = np.array([pinch_gt, clench_gt, poke_gt, palm_gt])
    gt_array = np.transpose(gt_array)
    pred_array = np.array([pinch_pred, clench_pred, poke_pred, palm_pred])
    pred_array = np.transpose(pred_array)
    mse, corr, acc = score_evaluation_from_np_batches(gt_array, pred_array)
    mse_list.append(mse)
    corr_list.append(corr)
    acc_list.append(acc)

for_save = {'mse':mse_list, 'corr':corr_list, 'acc':acc_list}
json.dump(for_save, open("lower_bound.json", 'w'))


##======================================================================
# predicting average scores from average of one-dropped-out scores

gt = {}
for key, val in labels.items():
    pi = [v for v in val['pinch'].values()]
    c = [v for v in val['clench'].values()]
    po = [v for v in val['poke'].values()]
    pa = [v for v in val['palm'].values()]
    f_ = [v for v in val['familiarity'].values()]
    f = []
    for e in f_:
        f += e

    piv = np.average(pi)
    cv = np.average(c)
    pov = np.average(po)
    pav = np.average(pa)
    fv = np.average(f)
    val = {'pinch': piv, 'clench': cv, 'poke': pov, 'palm': pav, 'familiarity': fv}
    gt[key] = val

mse_list = []
corr_list = []
acc_list = []
for _ in range(1000):
    out = {}
    for key, val in labels.items():
        pi = [v for v in val['pinch'].values()]
        c = [v for v in val['clench'].values()]
        po = [v for v in val['poke'].values()]
        pa = [v for v in val['palm'].values()]
        f_ = [v for v in val['familiarity'].values()]
        f = []
        for e in f_:
            f += e

        rpi = random.choice(pi)
        piv = (np.sum(pi) - rpi)/float(len(pi)-1)
        rc = random.choice(c)
        cv = (np.sum(c) - rc)/float(len(c)-1)
        rpo = random.choice(po)
        pov = (np.sum(po) - rpo)/float(len(po)-1)
        rpa = random.choice(pa)
        pav = (np.sum(pa) - rpa)/float(len(pa)-1)
        rf = random.choice(f)
        fv = (np.sum(f) - rf)/float(len(f)-1)
        val = {'pinch': piv, 'clench': cv, 'poke': pov, 'palm': pav, 'familiarity': fv}
        out[key] = val

    pinch_gt = []
    pinch_pred = []
    clench_gt = []
    clench_pred = []
    poke_gt = []
    poke_pred = []
    palm_gt = []
    palm_pred = []
    for k in out.keys():
        pinch_gt.append(gt[k]['pinch'])
        pinch_pred.append(out[k]['pinch'])
        clench_gt.append(gt[k]['clench'])
        clench_pred.append(out[k]['clench'])
        poke_gt.append(gt[k]['poke'])
        poke_pred.append(out[k]['poke'])
        palm_gt.append(gt[k]['palm'])
        palm_pred.append(out[k]['palm'])

    gt_array = np.array([pinch_gt, clench_gt, poke_gt, palm_gt])
    gt_array = np.transpose(gt_array)
    pred_array = np.array([pinch_pred, clench_pred, poke_pred, palm_pred])
    pred_array = np.transpose(pred_array)
    mse, corr, acc = score_evaluation_from_np_batches(gt_array, pred_array)
    mse_list.append(mse)
    corr_list.append(corr)
    acc_list.append(acc)

for_save = {'mse':mse_list, 'corr':corr_list, 'acc':acc_list}
json.dump(for_save, open("upper_bound1.json", 'w'))

##======================================================================
# predicting average of one half from average of the other half

gt = {}

mse_list = []
corr_list = []
acc_list = []
for _ in range(1000):
    out = {}
    for key, val in labels.items():
        pi = [v for v in val['pinch'].values()]
        c = [v for v in val['clench'].values()]
        po = [v for v in val['poke'].values()]
        pa = [v for v in val['palm'].values()]
        f_ = [v for v in val['familiarity'].values()]
        f = []
        for e in f_:
            f += e

        np.random.shuffle(pi)
        pi1 = pi[:int(len(pi)/2)]
        pi2 = pi[int(len(pi)/2):]
        piv = np.average(pi1)
        rpi = np.average(pi2)

        np.random.shuffle(c)
        c1 = c[:int(len(c)/2)]
        c2 = c[int(len(c)/2):]
        cv = np.average(c1)
        rc = np.average(c2)

        np.random.shuffle(po)
        po1 = po[:int(len(po)/2)]
        po2 = po[int(len(po)/2):]
        pov = np.average(po1)
        rpo = np.average(po2)

        np.random.shuffle(pa)
        pa1 = pa[:int(len(pa)/2)]
        pa2 = pa[int(len(pa)/2):]
        pav = np.average(pa1)
        rpa = np.average(pa2)

        np.random.shuffle(f)
        f1 = f[:int(len(f)/2)]
        f2 = f[int(len(f)/2):]
        fv = np.average(f1)
        rf = np.average(f2)

        val = {'pinch': piv, 'clench': cv, 'poke': pov, 'palm': pav, 'familiarity': fv}
        out[key] = val
        val = {'pinch': rpi, 'clench': rc, 'poke': rpo, 'palm': rpa, 'familiarity': rf}
        gt[key] = val

    pinch_gt = []
    pinch_pred = []
    clench_gt = []
    clench_pred = []
    poke_gt = []
    poke_pred = []
    palm_gt = []
    palm_pred = []
    for k in out.keys():
        pinch_gt.append(gt[k]['pinch'])
        pinch_pred.append(out[k]['pinch'])
        clench_gt.append(gt[k]['clench'])
        clench_pred.append(out[k]['clench'])
        poke_gt.append(gt[k]['poke'])
        poke_pred.append(out[k]['poke'])
        palm_gt.append(gt[k]['palm'])
        palm_pred.append(out[k]['palm'])

    gt_array = np.array([pinch_gt, clench_gt, poke_gt, palm_gt])
    gt_array = np.transpose(gt_array)
    pred_array = np.array([pinch_pred, clench_pred, poke_pred, palm_pred])
    pred_array = np.transpose(pred_array)
    mse, corr, acc = score_evaluation_from_np_batches(gt_array, pred_array)
    mse_list.append(mse)
    corr_list.append(corr)
    acc_list.append(acc)

for_save = {'mse':mse_list, 'corr':corr_list, 'acc':acc_list}
json.dump(for_save, open("upper_bound2.json", 'w'))


##======================================================================
# predicting single score from average of the others

import sys
sys.path.insert(1, '../ee148-project/')
import random
from scipy.stats import pearsonr
from evaluation import *
import numpy as np

gt = {}

mse_list = []
corr_list = []
acc_list = []
for _ in range(1000):
    out = {}
    for key, val in labels.items():
        pi = [v for v in val['pinch'].values()]
        c = [v for v in val['clench'].values()]
        po = [v for v in val['poke'].values()]
        pa = [v for v in val['palm'].values()]
        f_ = [v for v in val['familiarity'].values()]
        f = []
        for e in f_:
            f += e

        rpi = random.choice(pi)
        piv = (np.sum(pi) - rpi)/float(len(pi)-1)
        rc = random.choice(c)
        cv = (np.sum(c) - rc)/float(len(c)-1)
        rpo = random.choice(po)
        pov = (np.sum(po) - rpo)/float(len(po)-1)
        rpa = random.choice(pa)
        pav = (np.sum(pa) - rpa)/float(len(pa)-1)
        rf = random.choice(f)
        fv = (np.sum(f) - rf)/float(len(f)-1)
        val = {'pinch': piv, 'clench': cv, 'poke': pov, 'palm': pav, 'familiarity': fv}
        out[key] = val
        val = {'pinch': rpi, 'clench': rc, 'poke': rpo, 'palm': rpa, 'familiarity': rf}
        gt[key] = val

    pinch_gt = []
    pinch_pred = []
    clench_gt = []
    clench_pred = []
    poke_gt = []
    poke_pred = []
    palm_gt = []
    palm_pred = []
    for k in out.keys():
        pinch_gt.append(gt[k]['pinch'])
        pinch_pred.append(out[k]['pinch'])
        clench_gt.append(gt[k]['clench'])
        clench_pred.append(out[k]['clench'])
        poke_gt.append(gt[k]['poke'])
        poke_pred.append(out[k]['poke'])
        palm_gt.append(gt[k]['palm'])
        palm_pred.append(out[k]['palm'])

    gt_array = np.array([pinch_gt, clench_gt, poke_gt, palm_gt])
    gt_array = np.transpose(gt_array)
    pred_array = np.array([pinch_pred, clench_pred, poke_pred, palm_pred])
    pred_array = np.transpose(pred_array)
    mse, corr, acc = score_evaluation_from_np_batches(gt_array, pred_array)
    mse_list.append(mse)
    corr_list.append(corr)
    acc_list.append(acc)

for_save = {'mse':mse_list, 'corr':corr_list, 'acc':acc_list}
json.dump(for_save, open("upper_bound3.json", 'w'))
