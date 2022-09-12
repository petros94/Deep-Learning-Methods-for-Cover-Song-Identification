"""
Prediction Utils
"""
import torch
import numpy as np
from utils.generic import extract_frame_triplet

global device


def embed(model, frames):
    model.eval()
    with torch.no_grad():
        return model(frames)


def distance(model, frame1, frame2):
    e1 = embed(model, frame1)
    e2 = embed(model, frame2)
    return torch.norm(e1 - e2)


def triplet_frame_distance(model, frame1, frame2, frame3):
    e1 = embed(model, frame1)
    e2 = embed(model, frame2)
    e3 = embed(model, frame3)
    return torch.norm(e1 - e2), torch.norm(e1 - e3), torch.norm(e2 - e3)


def triplet_song_distance(model, song1, song2, song3):
    e1 = embed(model, song1)
    e2 = embed(model, song2)
    e3 = embed(model, song3)
    return torch.norm(e1 - e2), torch.norm(e1 - e3), torch.norm(e2 - e3)


def extract_and_distance(model, songs, triplets, triplet_idx, frame_idx):
    f1 = extract_frame_triplet(
        songs, triplets, model.frame_size, triplet_idx, 0, frame_idx, model.scale
    )
    f2 = extract_frame_triplet(
        songs, triplets, model.frame_size, triplet_idx, 1, frame_idx, model.scale
    )
    f3 = extract_frame_triplet(
        songs, triplets, model.frame_size, triplet_idx, 2, frame_idx, model.scale
    )

    f1 = torch.Tensor(np.array(f1)).unsqueeze().unsqueeze().to(device)
    f2 = torch.Tensor(np.array(f2)).unsqueeze().unsqueeze().to(device)
    f3 = torch.Tensor(np.array(f3)).unsqueeze().unsqueeze().to(device)

    return triplet_frame_distance(model, f1, f2, f3)
