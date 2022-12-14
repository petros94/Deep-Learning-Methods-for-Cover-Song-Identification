from utils.generic import segment_and_scale
import numpy as np

class SimpleDataset:
    def __init__(self, songs, scale=(1, 0.33)):
        print("Creating SimpleDataset")
        self.label_mapping = {k: i for i,k in enumerate(list(songs.keys()))}
    
        self.frames = []
        self.labels = []
        self.song_names = []
        for song_id, covers in songs.items():
            for cover in covers:
                repr = cover['repr']
                self.frames.append(segment_and_scale(repr, frame_size=None, scale=scale))
                self.labels.append(self.label_mapping[song_id])
                self.song_names.append(cover['cover_id'])

    def __len__(self):
        return len(self.song_names)

    def __getitem__(self, item):
        return self.frames[item], self.labels[item], self.song_names[item]
