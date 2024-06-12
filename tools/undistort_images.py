# Date: 2022.01.01
# Author: Ilia Shipachev, ilia@mosaic51.com
#
# Undistort and rectify set of images and output them with the same directory structure
# Uses calibration information as it provided in our legacy .yaml output format
#
# To use it in PTGui, load generated templated based on original .yaml file
# Set camera model to rectiliniear , set (a, b, c) and shifts to zero
# Change focal length according output of this script

import cv2 as cv
from pathlib import Path
import numpy as np
import re
from collections import defaultdict
import argparse
import sys
import time
import multiprocessing
from functools import partial
import json

import yaml

# TARGET_F = 1400   #for M51
TARGET_F = 1900   #for MX
# TARGET_F = 3000 #for viking

SENSOR_DIAG_MM = 43.2666 #as a default value in PTGui

#if both inputs and output are expected to be rotated to portrait orientation relative to default orientation stored in calibration .yaml
IS_ROTATED = False

# CPU_COUNT = multiprocessing.cpu_count() - 1
CPU_COUNT = 12 #having it higher may actually make system works slower, experiment with it, depends on the system and input size
CHUNKSIZE = 32 #emperically good chunk size for Ryzen 5950X and this task


def undistort_img(img_p: Path, out_p:Path, maps: dict):
    '''
    Reads image, undistor it and write it back

    Mapping selection done by finding camera module name (cam<N>)
    and matching with available keys in maps
    (for multiprocessing purposes)
    '''

    match = re.search(R"(?P<camname>cam\d+)", str(img_p))
    assert len(match.groups()) == 1, f"Found more than one 'cam' in {str(img_p)} or didn't find it at all"
    camname = match.group("camname")

    map1, map2 = maps[camname]
    img = cv.imread(str(img_p))

    und_img = cv.remap(img, map1, map2, interpolation=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)

    cv.imwrite(str(out_p), und_img)
    print(f"Using {camname} calib; {img_p} -->  {out_p}")


def undistort_imgs(src_p:Path, res_p:Path, calib_p:Path):
    '''
    Undistorts images read from src_p and grouped in directories
    per camera module, using yaml calibration file from chain_p
    or calibration in riegl compatible format from .json file
    '''

    with open(calib_p, 'rb') as file:
        if calib_p.suffix == ".yaml":
            cams = yaml.load(file, Loader=yaml.SafeLoader)
        elif calib_p.suffix == ".json":
            calib = json.load(file)
            if "cams" in calib:
                cams = calib["cams"]
            else:
                Exception("Non Riegl .jsons are not supported yet by this script")
        elif calib_p.suffix == ".grp":
            print("Only .json or .yaml calibration files can be used for undistortion, .grp in not applyable here")
            exit()
        print(cams)

    res_p.mkdir(parents=False, exist_ok=True)

    #We suppose that all cameras has the same resolution
    calib_dim = np.array(cams["cam0"]["resolution"], dtype=int)

    if not IS_ROTATED:
        target_dim = calib_dim
    else:
        target_dim = (calib_dim[1], calib_dim[0])

    #let's normalize to have optical center to the center of undistorted image
    target_cx  = target_dim[0] / 2 + 0.5
    target_cy  = target_dim[1] / 2 + 0.5 
    K_target   = np.array([[TARGET_F,       0,    target_cx],
                            [0,        TARGET_F,  target_cy],
                            [0,               0,          1]], 
                            dtype=np.float64)
    
    maps = dict.fromkeys(cams.keys())
    for camname in maps.keys():
        I = cams[camname]['intrinsics']
        fx, fy, cx, cy = I
        if IS_ROTATED:
            fx, fy = fy, fx
            cx, cy = cy, cx
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        D = np.array(cams[camname]['distortion_coeffs'])
        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_target,
                                                        target_dim, cv.CV_32FC1)
        maps[camname] = (map1, map2)


    #Preparing input paths and output paths
    imgs = sorted(list(src_p.rglob("*.jpg")) + list(src_p.rglob("*.png")))
    out_imgs = [res_p / img_p.relative_to(src_p) for img_p in imgs]
    for out_img_p in out_imgs:
        out_img_p.parent.mkdir(exist_ok=True, parents=True)

    st = time.time()
    print("CPU count: ", CPU_COUNT)
    with multiprocessing.Pool(CPU_COUNT) as pool:
        # pool.starmap(partial(undistort_img, maps=maps), zip(imgs, out_imgs), chunksize=40)
        pool.starmap(partial(undistort_img, maps=maps), zip(imgs, out_imgs), chunksize=CHUNKSIZE)
    et = time.time()
    res = et - st
    print('Execution time:', res, 'seconds')

    sensor_diag_px = np.linalg.norm(target_dim)
    pixel_mm = SENSOR_DIAG_MM / sensor_diag_px
    focal_mm = TARGET_F * pixel_mm

    print(f"Focal lenght (mm) relative to {SENSOR_DIAG_MM=}: {focal_mm}")
    print(f"Target focal lengh in pixesl: {TARGET_F}")
    print(f"Target principal point: {target_cx}, {target_cy}")
    print(f"You can change predefined target focal length and principal point in the top part of the script")

    return


def build_parser() -> argparse.ArgumentParser:
    ''' Prepare command line parser '''
    parser = argparse.ArgumentParser(description='Run simultaniosly all captured videos\
                                    take snapshots(screenshot) and capture all frames from \
                                    all cameras in the folder. By default script will create folder')
    parser.add_argument(
        "src",
        help="Input directory with set of extracted frames, in separate folders per each camera module",
        type=Path,
    )
    parser.add_argument(
        "dst",
        help="Output directory where undistorted frames will be stored",
        type=Path,
    )
    parser.add_argument(
        "camchain",
        help="Path to calibration file in camchain.yaml format",
        type=Path,
    )
    return parser

def main(args):

    if len(args) > 0:
        parser = build_parser()
        pargs = parser.parse_args(args)
        undistort_imgs(pargs.src, pargs.dst, pargs.camchain)
        
    #Debug option when not arguments were provided, you can comment it out if you don't need to use it this way
    else:
        print("Running without arguments in debug mode. Type --help to get available parameters. Will start in 3 secs")
        time.sleep(3)
        
        src_p = Path("/media/tomas/samQVO_4TB_D/asset-detection-datasets/drtinova_med_u_track/data_m/reel_0050_20231107-122212_prague_MX_XVN")
        dst_p = Path("/home/tomas/")
        calib_p = src_p/'calib.yaml'

        undistort_imgs(src_p, dst_p, calib_p)

if __name__ == "__main__":
    main(sys.argv[1:])