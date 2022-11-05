from utils.generic import segment_and_scale

class SimpleDataset:
    def __init__(self, songs, scale=(1, 0.33)):
        print("Creating SimpleDataset")
        self.label_mapping = {k: i for i,k in enumerate(list(songs.keys()))}
    
        self.frames = []
        self.labels = []
        for song_id, covers in songs.items():
            for cover in covers:
                repr = cover['repr']
                self.frames.append(segment_and_scale(repr, frame_size=None, scale=scale))
                self.labels.append(self.label_mapping[song_id])