import os 
from argparse import ArgumentParser
import subprocess
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser(description="Configuring param for training nerf")
    parser.add_argument("--img_path", type=Path,
                        help="Path to the image folder")
    parser.add_argument("--dataset", type=str,
                        help="dataset name")
    parser.add_argument("--scene", type=str,
                        help="scene name")
    parser.add_argument("--model", type=str,
                        default="nerfacto",
                        help="nerf model for training, default is nerfacto")
    parser.add_argument("--data_type", type=str,
                        default="instant-ngp-data",
                        help="data type for training, default is instant-ngp format")
    parser.add_argument("--eval_mode", type=str,
                        default="fraction")
    parser.add_argument("--train_split_fraction", type=float,
                        default=1.0)
    parser.add_argument("--n_iters", type=int,
                        default=60000,
                        help="number of training iteration")
    parser.add_argument("--output_path", type=str,
                        default="../outputs")

    args = parser.parse_args() 
    data_path = os.path.join(args.img_path, 
                             args.dataset,
                             args.scene,
                             "transform.json")
    os.chdir(args.output_path)

    subprocess.call([
        'ns-train', args.model,
        '--max-num-iterations', args.n_iters,
        args.data_type,
        '--data', data_path,
        '--eval-mode', args.eval_mode,
        '--train-split-fraction', args.train_split_fraction
        ])

