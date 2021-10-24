"""
Convert .pt weight file such that it works with relative paths rather than absolute ones
"""

import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov4-p5.pt', help='weights path')
    parser.add_argument('--outdir', type=str, default='', help='output path')
    opt = parser.parse_args()
    if opt.outdir == '':
        opt.outdir = opt.weights.replace('.pt', '_stateDict.pt')

    # ckpt = torch.load(opt.weights, map_location=torch.device('cpu')) # Load checkpoint
    # ckpt['model'] = ckpt['model'].float() # To float
    # torch.save(ckpt, opt.weights.replace('.pt', '_converted.pt')) # Save model

    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float() # Load checkpoint model
    stateDict = model.state_dict() # Retrieve state dict
    torch.save(stateDict, opt.outdir) # Save state dict
