import subprocess
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Optional
import argparse
#from nerfstudio.utils.rich_utils import CONSOLE
from rich import print as rprint
from utils.colmap2ngp import convert_colmap_to_ngp
from utils.misc import read_model, convert_bin2txt
from utils.hloc_processing import copy_images
import time
import io
import selectors

BIN_FILES = ["cameras.bin", "images.bin", "points3D.bin"]
TXT_FILES = ["cameras.txt", "images.txt", "points3D.txt"]
AABB_SCALE = 1.0
SKIP_EARLY = 0
KEEP_COLMAP_COORDS = True
TRAINING_ITERS = 40000

def extract_fnames_from_colmap(colmap_images):
    fnames = []
    for id_, image in colmap_images.items():
        fnames.append(image.name)
    return fnames

def capture_subprocess_output(subprocess_args):
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = subprocess.Popen(subprocess_args,
                                shell=True,
                               bufsize=1,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    # Create callback function for process output
    buf = io.StringIO()
    def handle_output(stream, mask):
        # Because the process' output is line buffered, there's only ever one
        # line to read when this function is called
        line = stream.readline()
        buf.write(line)
        sys.stdout.write(line)

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = (return_code == 0)

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)

def preprocessing(src_img_folder:str,
                  dst_img_folder:str,
                  sfm_model_path:str,
                  output_path:str,
                  dataset:str,
                  scene:str):
    # Copy images to be trained on to a new folder
    # Convert colmap format to transform.json
    rprint("[bold green]Get list of training images")
    cameras, images, points3D = read_model(os.path.join(sfm_model_path, dataset, scene, "sfm_superpoint+superglue"))
    training_fnames = extract_fnames_from_colmap(images)
    rprint(f"[bold green]{len(training_fnames)} images are used for training")
    rprint("[bold green]Copying images to new folder")
    output_folder = os.path.join(dst_img_folder, dataset, scene)
    if dataset == "12scenes":
        copy_images(os.path.join(src_img_folder, scene), output_folder, training_fnames)
    else:
        copy_images(os.path.join(src_img_folder, dataset, scene), output_folder, training_fnames)
    rprint("[bold green]Images are copied to new folder")
    # Check if the colmap model is binary or txt
    scene_path = os.path.join(sfm_model_path, dataset, scene, "sfm_superpoint+superglue")
    sfm_files = os.listdir(scene_path)
    if all([f in sfm_files for f in BIN_FILES]):
        rprint("[bold green]Colmap model is binary, converting to txt")
        convert_bin2txt(scene_path, scene_path)
    else:
        rprint("[bold green]Colmap model is already in txt, skip conversion")

    rprint("[bold green]Converting colmap model to instant-ngp transform.json")
    #instant_ngp_output = os.path.join(output_path, f"instant_ngp/training/{dataset}/{scene}/transform.json")
    transfrom_output = os.path.join(dst_img_folder, dataset, scene, "transform.json")
    convert_colmap_to_ngp(scene_path, 
                          output_folder,
                          out_path=transfrom_output,
                          keep_colmap_coords=KEEP_COLMAP_COORDS,
                          skip_early=SKIP_EARLY,
                          aabb_scale=AABB_SCALE)



def training(instant_ngp_dir:str, 
             data_path:str,
             dataset:str, 
             scene:str, 
             output_path:str):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Convert colmap model to instant-ngp transform.json
    #instant_ngp_output = os.path.join(output_path, f"instant_ngp/training/{dataset}/{scene}")
    
    # Run instant-ngp training
    rprint("[bold green]Running instant-ngp training")
    rprint("[bold green]Make sure that you are in the environment with instant-ngp installed")
    instant_ngp_command = f"python {os.path.join(instant_ngp_dir, 'scripts/run.py')}" + \
                            f" --scene={os.path.join(data_path, dataset, scene, 'transform.json')}" + \
                            f" --mode=nerf" + \
                            f" --save_snapshot={os.path.join(output_path, f'instant_ngp/snapshots/{dataset}/{scene}/instant_ngp_{timestr}.msgpack')}" + \
                            f" --n_steps={TRAINING_ITERS}"
    capture_subprocess_output(instant_ngp_command)
    #run_command(instant_ngp_command)



def view_synthesis(model_path:str, 
                   dataset:str, 
                   scene:str, 
                   output_path:str, 
                   sceenshot_transform:str):
    # Run instant-ngp view synthesis
    rprint("[bold green]Running instant-ngp view synthesis")
    rprint("[bold green]Make sure that you are in the environment with instant-ngp installed")
    output_folder = os.path.join(output_path, f"instant_ngp/view_synthesis/{dataset}/{scene}/")
    instant_ngp_command = f"python run.py" + \
                            f" --load_snapshot={model_path}" + \
                            f" --screenshot_transform={sceenshot_transform}" + \
                            f" --screenshot_dir={output_folder}" + \
                            f" --screenshot_w=640" + \
                            f" --screenshot_h=480"
    
    #run_command(instant_ngp_command)
    capture_subprocess_output(instant_ngp_command)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run instant-ngp")
    parser.add_argument("--mode", type=str, help="Mode to run instant-ngp")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument("--sfm", type=str, help="Path to sfm model")
    parser.add_argument("--filtered_sfm", type=str, help="Path to filtered sfm model")
    parser.add_argument("--src_img", type=str, help="Path to dataset images")
    parser.add_argument("--dst_img", type=str, help="Path to copy destination")
    parser.add_argument("--instant_ngp", type=str, help="Path to instant-ngp")
    parser.add_argument("--output", type=str, help="Path to output folder")
    parser.add_argument("--screenshot_transform", type=str, help="Path to screenshot transform")
    args = parser.parse_args()

    args.mode = "preprocessing"
    args.src_img = "/home/hoang/Hoang_workspace/dataset/datasets/"
    # args.src_img = "/media/hoang/Data/data/datasets/"
    #args.src_img = "/media/hoang/Data/data/12scenes/12scenes/12scenes_sfm_triangulated"
    args.dst_img = "/media/hoang/Data/D2S--/data/images/5_evenly"
    args.sfm = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_5_evenly"
    args.output = "/media/hoang/Data/D2S--/models"
    args.dataset = "7scenes"
    args.scene = "stairs"
    args.instant_ngp = "/home/hoang/Hoang_workspace/instant-ngp"

    # Preprocessing
    if args.mode == "preprocessing":
        preprocessing(args.src_img, 
                      args.dst_img, 
                      args.sfm, 
                      args.output, 
                      args.dataset, 
                      args.scene)

    elif args.mode == "training":
        training(args.instant_ngp, args.dst_img, args.dataset, args.scene, args.output)


if __name__ == "__main__":
    main()


    

    

    



    
    
