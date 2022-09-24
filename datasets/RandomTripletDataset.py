import imp
import torch
from utils.generic import generate_triplets, get_device, retrieve_repr, repr_triplet_2_segments, frame_idx_2_time

class RandomTripletDataset(torch.utils.data.Dataset):
    def __init__(self, songs, samples_per_song=10, frame_size=400, scale=(1, 0.33)):
        self.triplets = generate_triplets(songs, samples_per_song)
        self.songs = songs
        self.samples_per_song = samples_per_song
        self.frame_size = frame_size
        self.scale = scale
        
    def collate_fn(self, batch):
        x = []
        metadata = {
            "title": [],
            "time": [],
            "frame_id": [],
            "n_frames": 0
        }
        
        for triplet in batch:
            mfccs = [torch.from_numpy(retrieve_repr(self.songs, v['song_id'], v['cover_id'])) for v in list(triplet.values())]
            segs = repr_triplet_2_segments(mfccs, self.frame_size, scale=self.scale)
            x.append(segs)
            metadata["n_frames"] += segs.size(0)

        new_title = ["c_id: " + list(triplet.values())[0]['cover_id'] + " | s_id: " + list(triplet.values())[0]['song_id']]*len(segs)
        
        metadata['title'].extend(new_title)
        metadata['frame_id'].extend([i for i in range(len(segs))])

        for idx in range(len(segs)):
            time_start, time_end, duration = frame_idx_2_time(idx, self.frame_size)
            metadata['time'].append("start: {}, end: {}".format(time_start, time_end))
    
        return torch.cat(x).transpose(0,1).to(get_device()), metadata

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return triplet
    
    def __len__(self):
        return len(self.triplets)