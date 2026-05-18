import numpy as np
import cv2

def normalize_intensity(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-8)

def resize_frame(frame, size=224):
    return cv2.resize(frame, (size, size))

def temporal_sampling(volume, num_frames=20):
    total_frames = volume.shape[-1]
    indices = np.linspace(0, total_frames-1, num_frames).astype(int)
    return volume[..., indices]

def preprocess_volume(volume):
    
    mid_slice = volume.shape[2] // 2
    cine = volume[:, :, mid_slice, :]
    
    cine = temporal_sampling(cine)
    
    frames = []
    for t in range(cine.shape[-1]):
        frame = cine[:, :, t]
        frame = normalize_intensity(frame)
        frame = resize_frame(frame)
        frames.append(frame)
    
    frames = np.stack(frames) 
    frames = np.expand_dims(frames, axis=1) 
    
    return frames