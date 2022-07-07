import urllib.request
from html_table_parser.parser import HTMLTableParser
import clip
from utils import clip_model
from utils import device
import torch
from os.path import join as ospj
import json
import numpy as np
import util.config as cfg

def url_get_contents(url):
    """ Opens a website and read its binary contents (HTTP Response Body) """
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    return f.read()

def clip_embedding(text):
    text_token = clip.tokenize(text).to(device)
    with torch.no_grad():
        encoded_text = clip_model.encode_text(text_token)

    return encoded_text

def get_rawLabel(ann):
    # Get raw labels if they exist
    raw_l = []
    if ann['frame_ann'] is not None:
        # frame_l = flatten([seg['act_cat'] for seg in ann['frame_ann']['labels']])
        raw_l = [seg['raw_label']  for seg in ann['frame_ann']['labels'] if seg['raw_label']]
    return  raw_l

def labelFromJson(path):
    l_babel_files = ['train', 'val']

    babel = {}
    babel_raw_labels = []

    for file in l_babel_files:
        babel[file] = json.load(open(ospj(path, file + '.json')))
    for spl in babel:
        for sid in babel[spl]:
            raw_l = get_rawLabel(babel[spl][sid])
            babel_raw_labels.extend(raw_l)

    babel_raw_labels = np.array(babel_raw_labels)
    sorted_babel_raw_labels = np.unique(babel_raw_labels)

    return sorted_babel_raw_labels

def matching(cr_label, babel_label):
    matched_raw_labels = []
    for seg_label in cr_label:
        matches= set(seg_label).intersection(set(babel_label))
        # for label in seg_label:
        #     if not label in babel_label:
        #         seg_label.remove(label)
        #     if label == 'interact with object use both hands and lean down':
        #         seg_label.remove(label)
        # matched_raw_labels.append(seg_label)
        matched_raw_labels.append(list(matches))

    return matched_raw_labels

def crawling(url):
    xhtml = url_get_contents(url).decode('utf-8')
    p = HTMLTableParser()
    p.feed(xhtml)
    data = p.tables[0][1:]
    act_cat = [tb_row[1] for tb_row in data]
    # cr_raw_label = np.array([tb_row[-1].split(', ') for tb_row in data])

    return act_cat


def main():
    babel_folder = cfg.babel_path
    # url = "https://babel.is.tue.mpg.de/actionsrawlabels.html"
    # cr_act_cat = crawling(url)

    prefix = 'a 3D rendering of '
    suffix = ' in unreal engine'

    babel_raw_label = labelFromJson(babel_folder)
    raw_affix = np.array([prefix+label+suffix for label in babel_raw_label])
    # matched_raw_label = matching(cr_raw_label, babel_raw_label)
    encoded_raw_label = clip_embedding(raw_affix).cpu()

    # for labels in matched_raw_label:
    #     total_label.append(labels)
    #     if len(labels):
    #         raw_l = clip_embedding(labels).cpu()
    #         embedded_raw_label.append(raw_l)
    #     else:
    #         embedded_raw_label.append(labels)

    en_raw_label = np.array(encoded_raw_label)


if __name__ == '__main__':
    main()


