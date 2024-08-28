import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from pathlib import Path
import typing as T
import numpy as np
import json
from PIL import Image

# from .models import *

CLASSES_P = "/home/tomas/traffi-signs-training/traffic_signs_features/total_data_CNN03_filt/info.json"


def run_inference(net:nn.Module, img:T.Union[torch.tensor, Path], 
                  in_img_size, softmax_norm=False):
    '''
    Get last layer outputs given model and 
    args:
        - in_img_size(height,width)
    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    net.eval()
    if isinstance(img, Path):
        img = read_image(str(img)).float()
    elif isinstance(img, Image.Image):
        img = TF.pil_to_tensor(img).float()

    # add dummy batch dim TODO: batching?
    img = img.unsqueeze(0)
    img = TF.resize(img, in_img_size).to(device)

    # normalize as done during training
    img = TF.normalize(img, (0.0,0.0,0.0), (255,255,255))
    
    out, embedding = net(img)
    out, embedding = out.squeeze(0), embedding.squeeze(0)
    
    if softmax_norm:
        return F.softmax(out), embedding
    else:
        return out, embedding    
    
def run_batched_inference(net:nn.Module, img_batch, in_img_size, id_2_label, softmax_norm=False, pred_labels=False):       
    '''
    Get last layer outputs given model for target input batch 
    args:
        - in_img_size(height,width)
        - img_batch list of Path or of Image or tensor
        - pred_labels: if True return also string label
    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    net.eval()
    
    imgs = torch.zeros([len(img_batch), 3, in_img_size[0], in_img_size[1]]) # actuall batched img tensor
    if isinstance(img_batch[0], Path):
        for i, img_p in enumerate(img_batch):
            im = read_image(str(img_p)).float()
            imgs[i,:,:,:] = TF.resize(im, in_img_size)
    
    elif isinstance(img_batch[0], Image.Image):
        for i, img in enumerate(img_batch):
            im = TF.pil_to_tensor(img).float()
            imgs[i,:,:,:] = TF.resize(im, in_img_size)
    else:
        imgs = img_batch
     
    # normalize as done during training
    imgs = TF.normalize(imgs, (0.0,0.0,0.0), (255,255,255)).to(device)
    
    out, embedding = net(imgs)
    
    if softmax_norm:
        out = F.softmax(out)
    
    if not pred_labels:
        return out, embedding
    else:
        labels = [id_2_label[torch.argmax(o).item()]+": "+ str(torch.max(o).item()) if softmax_norm 
                  else str(torch.max(F.softmax(o).item())) for o in out]
        return out, embedding, labels

def get_prediction(img, model, softmax=True):

    with open(CLASSES_P) as f:
        classes_data = json.load(f)
    
    id_2_label = classes_data['id_to_label']

    model.eval()
    out_vec, embeddings = run_inference(model, img, (128,128), softmax_norm=softmax)
    out_vec = out_vec.cpu().detach().numpy()

    pred_label = id_2_label[np.argmax(out_vec)]

    return out_vec, embeddings.detach().cpu().numpy(), f"{pred_label}: {np.max(out_vec):.2f}%"

if __name__=='__main__':
    img_p = Path("/home/tomas/traffi-signs-training/traffic_signs_features/Cropped-Traffic-Signs-1obj-27_07_2023/B28/B28_7.jpg")
    # model_p = "traffic_signs_features/resnet_tiny/ResnetTiny_epoch_6.pth"
    # model =  ModelTinyHruzBottleneck(bottleneck_size=128) # torch.load(str(model_p))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = torch.load(model_p)
    # pred = get_prediction(img_p, model)

    from models import ResnetTiny

    checkpoint_p = "/home/tomas/traffi-signs-training/tinyResnet_128x128/ResnetTiny_epoch_8.pth"
    resize_to = (128,128)


    model = ResnetTiny()
    checkpoint = torch.load(checkpoint_p)
    model.load_state_dict(checkpoint['model_state_dict'])

    pred = get_prediction(img_p, model)
    print(pred[2])

    print("out vector sum:", np.sum(pred[0]))