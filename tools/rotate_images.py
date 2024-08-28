### SIMPLE SCRIPT TO ROTATE DIRECTORIES OF JPG IMAGES (--reel)
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys

# from jpegtran import JPEGImage


def rotate_90_lossless(input_path, output_path):
    '''rotating using the:
        - https://github.com/jbaiter/jpegtran-cffi
        requires libjpeg8 with headers
    '''
    img = JPEGImage(str(input_path))
    img.rotate(90).save(str(output_path))

def rotate(input_path, output_path, quality=90):
    img = Image.open(input_path)

    rotated_img = img.rotate(90, expand=True)
    
    rotated_img.save(output_path, quality=quality)  # quality=keep --cannot rotate creates copy img that is not a jpeg anymore

def rotate_imgs(input_p:Path, output_p:Path):
    '''args: [input_p:Path, output_p:Path]'''

    output_p.mkdir(exist_ok=False, parents=True)

    imgs_ps = list(input_p.rglob('*.jpg'))
    
    for img_p in tqdm(imgs_ps):
        out_p = output_p/img_p.name
        rotate(img_p, out_p)
         
if __name__ == "__main__":
    if len(sys.argv) == 3:
        rotate_imgs(sys.argv[1:])
    else:    
        input_p = "/media/tomas/samQVO_4TB_D/asset-detection-datasets/drtinova_med_u_track/data_m/reel_undistorted"
        output_p = "/media/tomas/samQVO_4TB_D/asset-detection-datasets/drtinova_med_u_track/data_m/reel_rotated"

        paths = (Path(input_p), Path(output_p))
        rotate_imgs(*paths)