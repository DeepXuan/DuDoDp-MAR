import argparse
import os
import yaml

import numpy as np
import torch as th

from patch_diffusion import dist_util
from patch_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    add_dict_to_dict,
    dict_to_dict
)
from geometry.build_gemotry import initialization
from geometry.syndeeplesion_data import test_image
import imageio.v2 as imageio
from tqdm import tqdm
from torch_radon.radon import FanBeam 
from torchvision.transforms import Resize 

param = initialization()
angles = np.linspace(-np.pi/2*1, np.pi/2*3, 640)
radon = FanBeam(641, angles, 1075, 1075, param.param['su']/param.reso/641)

def main(parser):
    args = parser.parse_args()
    yaml_path = args.config

    # parser = argparse.ArgumentParser()
    with open(yaml_path) as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    args_dict = add_dict_to_dict(args_dict, model_and_diffusion_defaults())

    a = args_dict['a']
    n = args_dict['n']
    delta_y = args_dict['delta_y']
    save_dir = args_dict['save_dir']
    data_path = args_dict['data_path']
    inner_dir = args_dict['inner_dir']
    n_img = args_dict['num_test_image']
    n_mask = args_dict['num_test_mask']

    dist_util.setup_dist()

    model_names = args_dict['model_path'].split(",")
    models = []

    for model_name in model_names:
        model, diffusion = create_model_and_diffusion(
            **dict_to_dict(args_dict, model_and_diffusion_defaults().keys())
        )
        add_dict_to_argparser(parser, args_dict)
        args = parser.parse_args()
        model.load_state_dict(
            dist_util.load_state_dict(model_name, map_location="cpu")
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        models.append(model)

    if model.classifier_free and model.num_classes and args.guidance_scale != 1.0:
        model_fns = [diffusion.make_classifier_free_fn(model, args.guidance_scale) for model in models]

        def denoised_fn(x0):
            s = th.quantile(th.abs(x0).reshape([x0.shape[0], -1]), 0.995, dim=-1, interpolation='nearest')
            s = th.maximum(s, th.ones_like(s))
            s = s[:, None, None, None]
            x0 = x0.clamp(-s, s) / s
            return x0    
    else:
        model_fns = models
        denoised_fn = None

    os.makedirs(save_dir, exist_ok=True)
    
    args.batch_size = 1
    resize1 = Resize(416)
    resize2 = Resize(512)

    fp = lambda x: radon.forward(resize1(x)) * param.reso
    bp = lambda x: resize2(radon.backward(radon.filter_sinogram(x/param.reso)))

    for imag_idx in range(n_img): # 200 for all test data
        print(imag_idx)
        for mask_idx in tqdm(range(n_mask)): # 10
            Xma, XLI, Xgt, M, Sma, SLI, Sgt, Tr = test_image(data_path, imag_idx, mask_idx, inner_dir)
            M = resize2(M)
            _ = fp(Xgt) # Must use forward before calling backward or specify a volume
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = diffusion.p_mar_loop
            sample = sample_fn(
                model_fns,
                (args.batch_size, 1, args.image_size, args.image_size),
                Sma.view(1, 1, 640, 641),
                Tr.view(1, 1, 640, 641),
                fp,
                bp,
                (a, n, delta_y),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                denoised_fn=denoised_fn,
                device=dist_util.dev()
            )

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            # print(np.min(sample.cpu().numpy()), np.max(sample.cpu().numpy()))
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            imageio.imwrite(os.path.join(save_dir, '%03d_%03d.png'%(imag_idx, mask_idx)), (sample.squeeze().cpu().numpy() / 255. *65535.).astype(np.uint16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/MAR.yaml',
                        help='yaml file for configuration')
    main(parser)
