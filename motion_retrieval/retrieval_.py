from os.path import join as ospj
import json
from collections import *
import cfg as cfg
import random
import time
import numpy as np
import torch
from utils import device


def getAnns(seq, pose_label, anns):
    ann = {}
    if seq['frame_ann'] is not None:
        raw_l = np.array([seg['raw_label'] for seg in seq['frame_ann']['labels']])
        if pose_label in raw_l:
            ann['feat_p'] = seq['feat_p']
            idx = np.where(raw_l==pose_label)[0]
            ann['start_t'] = [seq['frame_ann']['labels'][i]['start_t'] for i in idx]
            ann['end_t'] = [seq['frame_ann']['labels'][i]['end_t'] for i in idx]
            anns[seq['babel_sid']].append(ann)


def loadJson(l_babel_files):
    babel = {}
    for file in l_babel_files:
        babel[file] = json.load(open(ospj(cfg.babel_path, file + '.json')))
    featp2fps = json.load(open(cfg.featp2fps_path))

    return babel, featp2fps


def retrieval_motion(pose_label):
    l_babel_files = ['train', 'val']
    babel, featp2fps = loadJson(l_babel_files)
    anns = defaultdict(list)
    for spl in babel:
        for sid in babel[spl]:
           getAnns(babel[spl][sid], pose_label, anns)
    sample_sid = random.choice(list(anns.keys()))

    babel_seq = anns[sample_sid][0]
    featp = babel_seq['feat_p']
    sample_idx = int(time.time() % len(babel_seq['start_t']))
    fps = featp2fps[featp]

    # for multi frame motion
    end_frame = round(babel_seq['end_t'][sample_idx] * fps)
    start_frame = round(babel_seq['start_t'][sample_idx] * fps)

    if (end_frame - start_frame) > 600:
        length = end_frame - start_frame
        mid_frame = start_frame + round(length/2)
        start_frame = mid_frame - 259
        end_frame = mid_frame + 300

    amass_seq = np.load(ospj(cfg.amass_path, featp))
    amass_p = torch.Tensor(amass_seq['poses'][start_frame:end_frame+1])

    body_pose = amass_p[:, 3:66].reshape(-1, 21*3).to(device)
    left_hand_pose = amass_p[:, 66:111].to(device)
    right_hand_pose = amass_p[:, 111:].to(device)

    return body_pose, right_hand_pose, left_hand_pose
