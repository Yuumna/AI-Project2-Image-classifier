import torch
import torch.autograd as Variable
import torch.utils.data as data
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import json
import argparse
from PIL import Image
from train import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--image',type=str,help='Image to predict')
parser.add_argument('--checkpoint',type=str,help='Model checkpoint to use when predicting')
parser.add_argument('--topk',type=int,help='Return top k predictions')
parser.add_argument('--labels',type=str,help='JSON file containing label names')
parser.add_argument('--gpu',action='store_true',help='Use GPU if available')

args, _ = parser.parse_known_args()

def process_image(image):
    if args.image:
        image=args.image
    
   
    image_loader=transforms.Compose([ 
                        transforms.Resize(254),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()])
    
    Pil_Image= Image.open(image)
    Pil_Image= image_loader(Pil_Image).float()
    
    np_image=np.array(Pil_Image)
    
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    np_image=(np.transpose(np_image,(1,2,0))-mean)/std
    np_image=np.transpose(np_image,(2,0,1))
    
    return np_image
	
def predict_model(image, checkpoint, topk=5, labels=''):
    
    if args.image:
        image=args.image
    
    if args.checkpoint:
        checkpoint=args.checkpoint
    
    if args.topk:
        topk=args.topk
    
    if args.labels:
        labels=args.labels
        
    if args.gpu:
        gpu=args.gpu
    
    checkpoint_dict=torch.load(checkpoint)
    arch= checkpoint_dict['arch']
    num_labels= len(checkpoint_dict['class_to_idx'])
    hidden_units= checkpoint_dict['hidden_units']
    
    model= load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)
    
    if gpu and torch.cuda.is_available():
        model.cuda()
    
    was_training = model.training
    model.eval()
    
    image=process_image(image)
    
    image=Variable(torch.FloatTensor(image), requires_grad=True)
    image=image.unsqueeze(0)
    
    if gpu and torch.cuda.is_available():
         image=image.cuda()
    
    result = model(image).topk(topk)
    
    if gpu and torch.cuda.is_available():
        probs=torch.nn.functional.softmax(result[0].data,dim=1).cpu().numpy()[0]
        classes= result[1].data.cpu().numpy[0]
        
    else:
        probs=torch.nn.functional.softmax(result[0].data,dim=1).cpu().numpy()[0]
        classes= result[1].data.cpu().numpy[0]
    
    if lables:
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)
        
        labels= list(cat_to_name.values())
        classes= [labels[x] for x in classes]
        
    model.train(mode=was_training)
    
    
    if args.image:
        print('Prediction and probabilities:', list(zip(classes, probs)))
    
    return probs, classes
	
if args.image and args.checkpoint:
    predict(args.image, args.checkpoint)