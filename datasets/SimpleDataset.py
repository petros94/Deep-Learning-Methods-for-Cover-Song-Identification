from utils.generic import segment_and_scale
import numpy as np

class SimpleDataset:
    def __init__(self, songs, scale=(1, 0.33)):
        print("Creating SimpleDataset")
        self.songs = self.filter_per_size(songs, 300)
        self.label_mapping = {k: i for i,k in enumerate(list(self.songs.keys()))}
        self.inv_mapping = {i: k for i,k in enumerate(list(self.songs.keys()))}
    
        self.frames = []
        self.labels = []
        self.song_names = []
        for song_id, covers in self.songs.items():
            for cover in covers:
                repr = cover['repr']
                self.frames.append(segment_and_scale(repr, frame_size=None, scale=scale))
                self.labels.append(self.label_mapping[song_id])
                self.song_names.append(cover['cover_id'])

    def __len__(self):
        return len(self.song_names)

    def __getitem__(self, item):
        return self.frames[item], self.labels[item], self.song_names[item]

    def idx_2_lab(self, idx):
        return self.inv_mapping[idx]

    def filter_per_size(self, songs, frame_size):
        output = {}
        for song_id, covers in songs.items():
            c = []
            for cover in covers:
                if np.array(cover['repr']).shape[-1] > frame_size:
                    c.append(cover)

            if len(c) > 1:
                output[song_id] = c
        return output
