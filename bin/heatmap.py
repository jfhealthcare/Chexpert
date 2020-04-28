import sys
import os
import argparse
import cv2
import logging
import time
import torch
import json
from easydict import EasyDict as edict
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from model.classifier import Classifier # noqa
from data.utils import transform # noqa
from utils.heatmaper import Heatmaper # noqa

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Heatmap')
parser.add_argument('model_file',
                    default=None,
                    metavar='MODEL_FILE',
                    type=str,
                    help="CKPT file to the trained model")
parser.add_argument('cfg_file',
                    default=None,
                    metavar='CFG_FILE',
                    type=str,
                    help="Model config file in json format")
parser.add_argument('txt_file',
                    default=None,
                    metavar='TXT_FILE',
                    type=str,
                    help="TXT file only have jpg file path")
parser.add_argument('plot_path',
                    default=None,
                    metavar='PLOT_PATH',
                    type=str,
                    help="Path to save the jpg")
parser.add_argument('--alpha',
                    default=0.2,
                    type=float,
                    help="Transparancy \
                     alpha of the heatmap, default 0.2")
parser.add_argument('--prefix',
                    default='none', type=str,
                    help="Which value \
                    to use as image name, cfg.train_classes or 'none', \
                    defaul 'none'")
parser.add_argument('--device_ids',
                    default='0',
                    type=str,
                    help="GPU indices comma separated, e.g. '0,1' ")


def run(args):
    cfg_file = args.cfg_file
    with open(cfg_file) as f:
        cfg = edict(json.load(f))
        model = Classifier(cfg)
    disease_classes = [
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Atelectasis',
        'Pleural Effusion'
    ]
    device_ids = list(map(int, args.device_ids.split(',')))
    # check device
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    # load model from ckpt file
    ckpt = torch.load(args.model_file, map_location=device)
    model = model.to(device).eval()
    model.load_state_dict(ckpt['state_dict'])
    # create plot folder
    if not os.path.exists(args.plot_path):
        os.mkdir(args.plot_path)
    # construct heatmap_cfg
    heatmaper = Heatmaper(args.alpha, args.prefix, cfg, model, device)
    assert args.prefix in ['none', *(disease_classes)]
    with open(args.txt_file) as f:
        for line in f:
            time_start = time.time()
            jpg_file = line.strip('\n')
            prefix, figure_data = heatmaper.gen_heatmap(jpg_file)
            bn = os.path.basename(jpg_file)
            save_file = '{}/{}{}'.format(args.plot_path, prefix, bn)
            assert cv2.imwrite(save_file, figure_data), "write failed!"
            time_spent = time.time() - time_start
            logging.info(
                '{}, {}, heatmap generated, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        jpg_file, time_spent))


def main():

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
