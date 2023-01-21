import imp
import torch
import random
import numpy as np
from utils.generic import sample_songs, segment_and_scale

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, songs, n_batches=256, songs_per_batch=64, frame_size=400, scale=(1, 0.33), online=False):
        print("Creating TripletDataset")
        self.n_batches = n_batches
        self.songs_per_batch = songs_per_batch
        
        self.songs = self.filter_per_size(songs, frame_size)
        self.online = online
        print(f"Initial songs: {len(songs)}, after filtering: {len(self.songs)}")
        
        """
        {
            "120345 (song_id)": torch.tensor of size num_segs X num_covers X num_channels X num_features X frame_size
        }
        """
        self.song_segs = {}
        self.int_mapping = {k: i for i,k in enumerate(list(self.songs.keys()))}
        to_be_deleted = []
        for song_id, covers in self.songs.items():
            segs = []
            for cover in covers:
                repr = cover['repr']
                frames = segment_and_scale(repr, frame_size=frame_size, scale=scale)
                segs.append(frames)
            
            # Find minimum length
            min_len = min(list(map(lambda i: len(i), segs)))

            # Crop to minimum length
            segs = [seg[: min_len] for seg in segs]
            
            # Size num_segs X num_covers X num_channels X num_features X frame_size
            try:
                ret = torch.stack(segs, dim=1)
                self.song_segs[song_id] = ret
            except Exception as e:
                print(e)
                to_be_deleted.append(song_id)

        for song_id in to_be_deleted:
            del self.songs[song_id]
            
        
        # Create batches
        # Each sample contains P songs of K covers each
        self.batches = []
        self.total_samples = 0
        for b in range(self.n_batches):
            samples = []
            labels = []
            song_ids = []
            P = sample_songs(self.songs, self.songs_per_batch).keys()
            
            for song_id in P:
                int_label = self.int_mapping[song_id]
                if self.online:
                    k = random.randint(0, len(self.song_segs[song_id])-1) #Select a random part of the songs
                    samples.append((song_id, k))
                    labels.extend([int_label]*self.song_segs[song_id][k].size(0))
                    song_ids.extend([song_id]*self.song_segs[song_id][k].size(0))
                    self.total_samples += self.song_segs[song_id][k].size(0)
                else:
                    K = random.choice(self.song_segs[song_id])
                    samples.append(K)
                    labels.extend([int_label]*K.size(0))
                    self.total_samples += K.size(0)

            labels = torch.tensor(labels)
            if not self.online:
                # Samples are now a tensor of size P*K X num_channels X num_features X frame_size
                samples = torch.cat(samples)
                assert samples.dim() == 4

            if self.online:
                self.batches.append((samples, labels, song_ids))
            else:
                self.batches.append((samples, labels))
            
        print(f"Total samples: {self.total_samples}")
        
    def __getitem__(self, idx):
        if self.online:
            samples, labels, song_ids = self.batches[idx]
            output = []
            for song_id, k in samples:
                K = self.song_segs[song_id][k]
                output.append(K)
            output = torch.cat(output)
            return output, labels, song_ids
        else:
            return self.batches[idx]
    
    def __len__(self):
        return len(self.batches)
    
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
                
                        
    # def get_full_songs(self):
    #     songs_repr = []
    #     labels = []
    #     for song_id, covers in self.songs.items():
    #         int_label = self.int_mapping[song_id]
    #         for cover in covers:
    #             repr = cover['repr']
    #             frames = segment_and_scale(repr, frame_size=None, scale=(1, 0.33))
    #             songs_repr.append(frames)
    #             labels.append(int_label)
                
    #     return songs_repr, labels