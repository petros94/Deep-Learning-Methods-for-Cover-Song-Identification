"""
Generic Utils
"""
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def retrieve_repr(songs, song_id, cover_id) -> np.array:
    """
    Retrieve the repr array for a given song and cover.
    Output array size: #num_features X num_samples
    """
    for s in songs[song_id]:
        if s["cover_id"] == cover_id:
            return s["repr"]


def sample_songs(songs, n_samples, ):
    """
    Return a subset of the songs dict, retured by load_songs()
    """
    if n_samples >= len(songs.keys()):
        print(f"n_samples exceeds dataset size of: {len(songs.keys())}. Full dataset will be used.")
        return songs
    
    keys = random.sample(sorted(songs.keys()), n_samples)
    return {k: songs[k] for k in keys}

def split_songs(songs, train_perc=0.8):
    """Split songs into train and test dict

    Args:
        songs (dict): the songs dict
        train_perc (float, optional): The train percentage. Defaults to 0.8.

    Returns:
        array of 2 dicts
    """
    items = list(songs.keys())
    X_train, X_test = train_test_split(items, train_size=train_perc)
    return {k: songs[k] for k in X_train}, {k: songs[k] for k in X_test}

def generate_triplets(songs, samples_per_song=10):
    """
    Return triplets of songs, in the following form:
    [
        {
            "anchor": {
                "song_id": "123456",
                "cover_id": "121212",
            },
            "pos_song": {
                "song_id": "123456",
                "cover_id": "131313",
            },
            "neg_song": {
                "song_id": "656533",
                "cover_id": "836664",
            },
        }
        ...
    ]

    Used for triplet-loss function
    """
    triplets = []
    for k in songs:
        for anchor in songs[k]:
            song_id, cover_id = anchor["song_id"], anchor["cover_id"]
            for n in range(samples_per_song):
                # pick random positive song
                while True:
                    pos_song = random.choice(songs[k])
                    if pos_song["cover_id"] != cover_id:
                        break

                # pick random negative song
                while True:
                    neg_songs = random.choice(list(songs.values()))
                    neg_song = random.choice(neg_songs)
                    if neg_song["song_id"] != song_id:
                        break

                # add to triplets
                triplets.append(
                    {
                        "anchor": {
                            "song_id": anchor["song_id"],
                            "cover_id": anchor["cover_id"],
                        },
                        "pos_song": {
                            "song_id": pos_song["song_id"],
                            "cover_id": pos_song["cover_id"],
                        },
                        "neg_song": {
                            "song_id": neg_song["song_id"],
                            "cover_id": neg_song["cover_id"],
                        },
                    }
                )
    return triplets


def generate_segments(song: np.array, step=400, overlap=0.5):
    """
    Segment a #num_features X num_samples vector into windows.

    Arguments:
        song: np.array or array like of shape num_features x num_samples
        step: the window_size
        overlap: the overlap percentage between windows

    To calculate the time is seconds for each window use the following formula:
    T = step * hop_size / sample_freq
    In the case of mfcc for example, T = step * 512 / 22050

    Returns a python list of shape num_segments X num_features X num_samples
    """
    return [
        song[..., i : step + i]
        for i in np.arange(0, song.shape[-1] - step, int(step * overlap))
    ]


def segment_and_scale(repr, frame_size, scale) -> torch.tensor:
    """
    Take an np.array of shape num_features x num_samples, split into segments and scale to specific size
    in order to create CNN inputs.
    
    Returns: num_segs X num_channels X num_features X num_samples
    """
    if type(repr) in (torch.tensor, np.ndarray):
        repr = torch.tensor(repr, dtype=float)
        
        if frame_size is None or repr.size(1) <= frame_size:
            frames = repr.unsqueeze(0)
        else:
            frames = torch.stack(generate_segments(repr, step=frame_size))
        frames = frames.unsqueeze(1)
        frames = F.interpolate(frames, scale_factor=scale)
        return frames
    
    elif type(repr) == list:
        scaled = [scale_dimensions_to_anchor(repr[0], r) for r in repr]        
        
        # num_channels X num_features X num_samples
        song_repr = torch.stack(scaled)
        # num_segs X num_channels X num_features X num_samples
        frames = torch.stack(generate_segments(song_repr, step=frame_size))
        frames = F.interpolate(frames, scale_factor=scale)
        
        assert frames.dim() == 4
        assert frames.size()[1] == len(repr)
        return frames
    else:
        raise ValueError("unsupported repr type")
        
        
def scale_dimensions_to_anchor(anchor, repr):
    repr = torch.tensor(repr)
    anchor = torch.tensor(anchor)
    
    anchor_repr_size = torch.tensor(anchor.size())
    current_repr_size = torch.tensor(repr.size())
    
    if (torch.all(anchor_repr_size == current_repr_size)):
        # No scale needed
        return repr
    
    output_size = (anchor_repr_size).tolist()
    repr = repr.unsqueeze(0).unsqueeze(0)
    repr = F.interpolate(repr, size=output_size)
    repr = repr.squeeze(0).squeeze(0)
    return repr

def repr_triplet_2_segments(triplet, frame_size, scale=(1, 1)):
    """
    Take a list of np.arrays, split them into segments and return a stacked torch tensor
    Arguments:
        triplet: list of np.arrays (usually 3 elements but it could work with N)

    Output:
        tensor of size num_segments X 3 X 1 X num_features X num_samples

        This is convenient for training because the num_segments is the number of individual samples,
        the "3" is used to easily split the anchor, pos and neg samples in case of triplet loss
        and the 1 X num_features X num_samples corresponds to the CNN input C X W X H.
    """
    segs = []
    for repr in triplet:
        frames = segment_and_scale(repr, frame_size=frame_size, scale=scale)
        segs.append(frames)

    # Find minimum length
    min_len = min(list(map(lambda i: len(i), segs)))

    # Crop to minimum length
    segs = [seg[: min_len - 1] for seg in segs]

    # zip samples
    (anchor, pos, neg) = segs
    ret = torch.stack((anchor, pos, neg), dim=1).unsqueeze(2)
    return ret


def frame_idx_2_time(frame_idx, frame_size, overlap=0.5):
    """
    Convenient utility to translate frame index to time.
    """
    overlap = 0.5 * frame_size
    time_start = frame_idx * overlap * 512 / 22050
    time_end = (frame_idx * overlap + frame_size) * 512 / 22050

    duration = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
    time_start = time.strftime("%H:%M:%S", time.gmtime(time_start))
    time_end = time.strftime("%H:%M:%S", time.gmtime(time_end))
    return time_start, time_end, duration


def extract_frame(songs, song_id, cover_id, frame_size, frame_idx, scale=(1, 1)):
    """
    Convert song to segments (frames) and extract a single frame
    """
    repr = retrieve_repr(songs, song_id, cover_id)
    segs = segment_and_scale(repr, frame_size=frame_size, scale=scale)
    frame = segs[frame_idx]
    return frame


def extract_frame_triplet(
    songs, triplets, frame_size, triplet_idx, song_idx, frame_idx, scale=(1, 1)
):
    s = list(triplets[triplet_idx].values())[song_idx]
    song_id = s["song_id"]
    cover_id = s["cover_id"]

    return extract_frame(songs, song_id, cover_id, frame_size, frame_idx, scale=scale)
