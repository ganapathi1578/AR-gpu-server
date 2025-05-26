import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import params
import torch
import numpy as np
import cv2
from torch.serialization import add_safe_globals
#from lib.slowfastnetworks import SlowFast
#from lib.slowfastnet import resnet50
import os
#sys.path.append('C:\Users\GANAPATHI\Desktop\NIT\project\iit_internship\action_recognition\actionrecapp')
#add_safe_globals({"SlowFast": SlowFast})
from lib.slowfastnet import SlowFast  # import the correct class used while saving

# Allow list the global class
from torch.serialization import add_safe_globals
add_safe_globals({"SlowFast": SlowFast})


def model_predict_lable_101(videotensor):
    # Step 1: Instantiate the model
    #model = resnet50(class_num=params['num_classes'])

    # Step 2: Load the weights only (no pickle issues)
    #state_dict = torch.load(params['model_path'])
    #model.load_state_dict(state_dict)
    
    model = torch.load(params['model_path'], weights_only=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    videotensor = videotensor.to(device)


    with open(params['labels_path'], 'r') as f:
        labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]
    with torch.no_grad():
            output = model(videotensor)
            probabilities = F.softmax(output, dim=1)
            max_idx = torch.argmax(probabilities, dim=1).item()
            pred_lable = labels[max_idx].lower()
    return pred_lable


def preprocess_video(video_path, clip_len, frame_sample_rate, crop_size):
    capture = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while len(frames) < clip_len and count < frame_count:
        ret, frame = capture.read()
        if not ret:
            break
        if count % frame_sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    capture.release()
    
    # If less frames than clip_len, pad by repeating last frame
    while len(frames) < clip_len:
        frames.append(frames[-1])
    
    # Convert list to np array: (clip_len, H, W, 3)
    frames = np.array(frames, dtype=np.float32)
    
    # Resize frames to crop_size x crop_size (center crop with resize)
    h, w, _ = frames[0].shape
    scale = crop_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    frames_resized = np.zeros((clip_len, new_h, new_w, 3), dtype=np.float32)
    for i in range(clip_len):
        frames_resized[i] = cv2.resize(frames[i], (new_w, new_h))
    
    # Center crop
    start_h = (new_h - crop_size) // 2
    start_w = (new_w - crop_size) // 2
    frames_cropped = frames_resized[:, start_h:start_h+crop_size, start_w:start_w+crop_size, :]
    
    # Normalize: (frame - 128) / 128
    frames_norm = (frames_cropped - 128.0) / 128.0
    
    # To tensor shape: (C, T, H, W)
    tensor = torch.tensor(frames_norm).permute(3, 0, 1, 2).unsqueeze(0)  # add batch dim
    
    return tensor


if __name__ == "__main__":
    num_classes = 101
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 64, 224, 224))
    #smodel = resnet50(class_num=num_classes)
    #output = model(input_tensor)
    #print(output.size())
