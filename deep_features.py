# USE traffic_signs_features to get deep features for img bbox corresponding to ray
import numpy as np
import torch
from PIL import Image, ImageDraw
import typing as T
from pathlib import Path
from tqdm import tqdm
import time
import json

from data_structures import Ray, Point
from traffic_signs_features.models import ResnetTiny
from traffic_signs_features.inference import run_inference, get_prediction, run_batched_inference


CHECKPOINT_P = "/home/tomas/traffi-signs-training/tinyResnet_128x128_merged_va_85/ResnetTiny_epoch_8.pth"
CLASSES_P = "/home/tomas/traffi-signs-training/traffic_signs_features/total_data_merged/info.json"

def get_id_to_label():
    with open(CLASSES_P) as f:
        classes_data = json.load(f)
    return classes_data['id_to_label']


# TODO: move to common
def get_frame_filename(sensor, frame_number, length=7, suffix='.jpg'):
    num_str = str(frame_number)
    if len(num_str) >= length:
        frame_id = num_str[-length:]
    else:
        frame_id = '0' * (length - len(num_str)) + num_str
    return f"{sensor}_frame_{(frame_id)}.jpg"

def get_deep_features(rays:T.List[Ray], debug=False):
    '''Returns vectors of cls_features and embeddings for each ray'''
    
    RESIZE_TO = (128,128)
    rotated_extracted_p = Path("/media/tomas/samQVO_4TB_D/drtinova_small_u_track/data_rotated")

    model = ResnetTiny()
    checkpoint = torch.load(CHECKPOINT_P)
    model.load_state_dict(checkpoint['model_state_dict'])


    EMB_SIZE = 8192 # (128 x 128) --> 4 x 4 x 512
    embeddings = np.zeros([len(rays), EMB_SIZE])
    cls_features = np.zeros([len(rays), 254])

    for i, r in enumerate(tqdm(rays)):
        img_p = rotated_extracted_p / get_frame_filename(r.sensor, r.frame_id)
        img = Image.open(img_p)
        crop_rect = (r.bbox[0], r.bbox[1], r.bbox[0]+r.bbox[2], r.bbox[1]+r.bbox[3])
        img = img.crop(crop_rect)

        out_pred, out_emb = run_inference(model, img, RESIZE_TO, softmax_norm=False)

        cls_features[i] = out_pred.detach().cpu().numpy()
        embeddings[i] = out_emb.detach().cpu().numpy()

        if debug:
            out_vec, pred_label = get_prediction(img, model)
            print(out_vec, f"sum: {np.sum(out_vec)}")
            # Draw the text on the image
            draw = ImageDraw.Draw(img)
            position = (2, 2 )
            draw.text(position, pred_label, fill=(0, 255, 0))
            img.show()

    return cls_features, embeddings


# not that faster --probably the Image loading and cropping? 
def get_deep_features_batched(rays:T.List[Ray], batch_size=64, softmax=True, num_classes=247, embeddings=False, debug=None,
                              rotated_extracted_p=Path("/media/tomas/samQVO_4TB_D/drtinova_small_u_track/data_rotated")):

    RESIZE_TO = (128,128)
    
    debug = Path(debug) if Path(debug).exists() else None
 
    model = ResnetTiny(num_out_classes=num_classes)
    checkpoint = torch.load(CHECKPOINT_P)
    model.load_state_dict(checkpoint['model_state_dict'])


    EMB_SIZE = 8192 # (128 x 128) --> 4 x 4 x 512
    
    embeddings = np.zeros([len(rays), EMB_SIZE]) if embeddings else None

    cls_features = np.zeros([len(rays), num_classes])

    # TODO: order by frame filename, so each frame opened only once

    rays_sorted = sorted(rays, key=lambda x: x.frame_id)
    
    stats = {'loading':0, 'inference': 0}
    
    for i in tqdm(range(len(rays)//batch_size+1)):
        # sliding window over in/out
        low_i = i*batch_size
        upp_i = (i+1)*batch_size
        
        if upp_i+1 > len(rays):
            upp_i = None # until the last (-1)

        # sorted not works here: order needed: TODO: implement argsort to fix it 
        ray_batch = rays[low_i:upp_i] # rays_sorted[low_i:upp_i]
        
        t_load = time.monotonic()
        img_batch = []
        filename = ""
        img = None
        for r in ray_batch:
            new_filename = get_frame_filename(r.sensor, r.frame_id)
            if filename != new_filename:
                filename = new_filename
                img = Image.open(rotated_extracted_p / filename)            

            img_batch.append(img.crop((r.bbox[0], r.bbox[1], r.bbox[0]+r.bbox[2], r.bbox[1]+r.bbox[3])))

            # if debug is not None: # run one image per batch aka superdumb extra inference
            #     # img = img_batch[0]
            #     out_vec, emb, pred_label = get_prediction(img, model, softmax=True) # 
            #     draw = ImageDraw.Draw(img)
            #     position = (2, 2)
            #     draw.text(position, pred_label, fill=(0, 255, 0))
            #     img_batch[0].save(f'{str(debug)}/p_{}_frame{ray_batch[0].frame_id}.png')

        stats['loading'] += time.monotonic() - t_load

        # inference
        t_inference = time.monotonic()
        cls_feat, emb_feat, labels = run_batched_inference(model, img_batch, RESIZE_TO, softmax_norm=softmax, pred_labels=True)
        stats['inference'] += time.monotonic() - t_inference
        cls_features[low_i:upp_i, :] = cls_feat.detach().cpu().numpy()
        if embeddings:
            embeddings[low_i:upp_i, :] = emb_feat.detach().cpu().numpy()


        
        
        if debug: 
            id_to_label = get_id_to_label()
            for j, feat in enumerate(cls_feat):
                lbl = id_to_label[torch.argmax(feat).item()] + f" {(torch.max(feat).item()):.2f}%"
                draw = ImageDraw.Draw(img_batch[j])
                position = (2, 2)
                draw.text(position, lbl, fill=(0, 255, 0))
                img_batch[j].save(f'{str(debug)}/point_{low_i+j}.png')

    return cls_features, embeddings, stats


def genereate_deep_features_midpoints(midpoints:T.List[Point], out_p:Path,
                                      batch_size=32, softmax=True, embedding_size=8192, num_classes=247,
                                      debug=None, return_labels=False, 
                                      data_path="/media/tomas/samQVO_4TB_D/drtinova_small_u_track/data_rotated"):
    ''' Same as get_deep_features_midpoints, but stores the matrices to the drive'''
    RESIZE_TO = (128,128)
    
    rotated_extracted_p = Path(data_path)

    model = ResnetTiny(num_out_classes=num_classes)
    checkpoint = torch.load(CHECKPOINT_P)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    cls_features = np.zeros((2, len(midpoints), num_classes))  # cls_features stored (embeddings quite bigger)
    
    for ridx, r_lbl in enumerate(['r2', 'r1']):
        rays = [getattr(point, r_lbl) for point in midpoints]
        if debug:
            debug_dir = Path(debug)/r_lbl
            debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            debug_dir = None

        feats, embeddings, stats = get_deep_features_batched(rays, batch_size=batch_size, softmax=softmax, num_classes=num_classes,
                                                             rotated_extracted_p=rotated_extracted_p, debug=debug_dir)
        cls_features[ridx,:,:] = feats
        
        # np.save(str(out_p/f"cls_features{r_lbl}"), cls_features)
        # np.save(str(out_p/f"embeddings{r_lbl}"), embeddings)

        print(f"{ridx}/2 deep features generated, stats: \n{stats}")

    return cls_features