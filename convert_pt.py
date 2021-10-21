"""
Convert .pt weight file such that it works with relative paths rather than absolute ones
"""

import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov4-p5.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[896, 896], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    ckpt = torch.load(opt.weights, map_location=torch.device('cpu')) # Load checkpoint
    ckpt['model'] = ckpt['model'].float() # To float
    torch.save(ckpt, opt.weights.replace('.pt', '_converted.pt')) # Save model
