import argparse
import os

from emotic import Emotic
from inference import inference_emotic

#https://analyticsindiamag.com/top-8-datasets-available-for-emotion-detection/
#http://sunai.uoc.edu/emotic/index.html
#https://github.com/rkosti/emotic
#https://github.com/Tandon-A/emotic

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_name', type=str, default='models', help='Name of the directory with the models')
    parser.add_argument('--config_dir_name', type=str, default='config', help='Name of the directory with the config files')
    parser.add_argument('--inference_file', type=str, help='Text file containing image context paths and bounding box')
    parser.add_argument('--context_model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='context model type')
    parser.add_argument('--body_model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='body model type')
    # Generate args
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model_path = 'face_models'

    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness', \
           'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    vad = ['Valence', 'Arousal', 'Dominance']
    ind2vad = {}
    for idx, continuous in enumerate(vad):
        ind2vad[idx] = continuous

    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]
    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]
    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]

    if args.inference_file is None:
        raise ValueError('Inference file not provided. Please pass a valid inference file for inference')
    pred_cats, pred_conts = inference_emotic(args.inference_file, model_path, context_norm, body_norm, ind2cat, ind2vad, args)
    for cat, cont in zip(pred_cats, pred_conts):
        print(cat, cont)


