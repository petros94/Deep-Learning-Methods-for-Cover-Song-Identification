
'''
Visualization Utils
'''

import numpy as np
import plotly.express as px
import time
from sklearn.decomposition import TruncatedSVD
import pandas as pd

from utils.generic import extract_frame, frame_idx_2_time

def visualize_frame_standalone(songs, song_id, cover_id, frame_idx, frame_size, scale=(1, 0.33)):
  frame = extract_frame(songs, song_id, cover_id, frame_size, frame_idx, scale)  
  time_start, time_end, duration = frame_idx_2_time(frame_idx, frame_size, 0.5*frame_size)

  print(f"Song id: {cover_id}, timeframe: {time_start} - {time_end}, duration: {duration}")
  f = px.imshow(frame, aspect='auto', range_color=[-3, 3], width=400, height=400)
  f.show()


def visualize_frame(songs, triplets, frame_size, triplet_idx, song_idx, frame_idx, scale=(1, 0.33)):
  s = list(triplets[triplet_idx].values())[song_idx]
  song_id = s['song_id']
  cover_id = s['cover_id']
  
  visualize_frame_standalone(songs, song_id, cover_id, frame_idx, frame_size, scale)


def visualize_frames(triplets, frame_size, triplet_idx, frame_idx):
  visualize_frame(triplets, frame_size, triplet_idx, 0, frame_idx)
  visualize_frame(triplets, frame_size, triplet_idx, 1, frame_idx)
  visualize_frame(triplets, frame_size, triplet_idx, 2, frame_idx)


def visualize_embeddings_pca(embeddings: np.array, 
                             extra_args: dict = None, 
                             colors: list = None,
                             dim: int = 2, 
                             dimensions: list = None):
  svd = TruncatedSVD(n_components=dim)
  svd.fit(embeddings)

  Y = svd.transform(embeddings)

  print(f"Explained variance: {svd.explained_variance_ratio_}")
  
  points = pd.DataFrame(Y)

  if extra_args is not None:
    for k in extra_args:
      points[k] = extra_args[k]

  points['color'] = colors

  fig = px.scatter_matrix(points, 
                          color='color',
                          dimensions=dimensions, 
                          custom_data=points.columns)
  
  hovertemplate = ""
  for idx, c in enumerate(points.columns.map(str)):
    hovertemplate += "<b>" + c + ": %{customdata[" + str(idx) + "]}</b><br>"

  hovertemplate += '<extra></extra>'

  fig.update_traces(hovertemplate=hovertemplate)
  fig.show()
  return points