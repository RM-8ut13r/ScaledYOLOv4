"""
Convert .pt weight file such that it works with relative paths rather than absolute ones
"""

import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov4-p5.pt', help='weights path')  # from yolov5/models/
    opt = parser.parse_args()

    ckpt = torch.load(opt.weights, map_location=torch.device('cpu')) # Load checkpoint
    ckpt['model'] = ckpt['model'].float() # To float
    torch.save(ckpt, opt.weights.replace('.pt', '_converted.pt')) # Save model
