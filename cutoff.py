import pickle
import os
import numpy as np


def chage_fp(model_dict):
    for k in model_dict['blobs'].keys():
        model_dict['blobs'][k] = model_dict['blobs'][k].astype(np.float16)

def chage_cls_fp(model_dict):
    for k in model_dict['blobs'].keys():
        if k.startswith('res'):
            model_dict['blobs'][k] = model_dict['blobs'][k].astype(np.float16)

def remove_momentum(model_dict):
    del_list = []
    for k in model_dict['blobs'].keys():
        if k.endswith('_momentum'):
            del_list.append(k)
    for i in del_list:
        del model_dict['blobs'][i]


def load_and_convert_coco_model(origin_model_path, target_model):
    with open(origin_model_path, 'rb') as f:
        model = pickle.load(f, encoding='latin1')
   
    chage_fp(model)
   
    remove_momentum(model)
    with open(target_model, 'wb') as f:
        pickle.dump(model, f, protocol=2)



if __name__ == '__main__':
    origin_model_path = "model_final.pkl" #replace with your model name
    target_model = origin_model_path.split('.')[0] + "_new.pkl"
    print("starting remove momentum, wait for a moment")
    load_and_convert_coco_model(origin_model_path, target_model)
    print("Done! Enjoy your competition")
