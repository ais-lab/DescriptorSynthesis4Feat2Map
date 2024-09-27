import os 
import json
import argparse
from pathlib import Path
import subprocess

def file_name_changes(camera_path, render_folder):
    transform = json.loads(open(camera_path).read())
    fnames = os.listdir(render_folder)
    fnames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    original_names = []
    for _, n in enumerate(transform["camera_path"]):
        original_names.append(n["file_path"])

    assert len(fnames) == len(original_names)
    #print(fnames[:5])
    #print(original_names[:5])
    # mapping
    file_map = {}
    for _, (nerfname, original_name) in enumerate(zip(fnames, original_names)):
        file_map[nerfname] = original_name.replace("/", "-")
    #print(file_map)

    # Traverse over reder folder and change the file name
    print("Renaming files to match the original names")
    for _, fname in enumerate(fnames):
        try:
            full_path = os.path.join(render_folder, fname)
            origin_name = os.path.join(render_folder, file_map[fname])
            os.rename(full_path, origin_name)
        except Exception as e:
            print(e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novel view synthesis using Nerfacto and nerfstudio cli")

    parser.add_argument("--mode", type=str,
                        default="camera-path")
    parser.add_argument("--dataset", type=str,
                        default='7scenes')
    parser.add_argument("--scene", type=str,
                        default="chess")
    parser.add_argument("--config", type=Path,
                        help="Path to the yaml file of the trained nerf model")
    parser.add_argument("--camera_path", type=Path,
                        help="Path to the synthesized camera path json")
    parser.add_argument("--rendered_output_names", type=str,
                        default="rgb")
    parser.add_argument("--output_format", type=str,
                        default="images")
    parser.add_argument("--output_path", type=Path,
                        help="Path to the output directory")
    parser.add_argument("--image_format", type=str,
                        default="png")

    args = parser.parse_args()

    command = ['ns-render']
    command.append(args.mode)
    command.append(['--load_config', args.config])
    command.append(['--rendered-output-names', 
                    args.rendered_output_names])
    command.append(['--camera-path-filename', 
                    args.camera_path])
    command.append(['--output-format', args.output_format])
    command.append(['--output-path'],
                   args.output_path / args.dataset / args.scene)
    command.append(['--image-format', args.image_format])

    results = subprocess.run(command, 
                             capture_output=True, 
                             text=True)

    print(results.stdout)
    print(results.stderr)






