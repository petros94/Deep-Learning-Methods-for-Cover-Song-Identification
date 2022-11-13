from feature_extraction.downloader import YoutubeDownloader
from feature_extraction.extraction import FeatureExtractor
import pandas as pd
import os
from scipy.io import savemat
from tqdm import tqdm

COL_NAMES = {
    "song_name": "Song name",
    "original_song": "YouTube URL - Original Song",
    "cover_1": "YouTube URL - Cover Song 1",
    "cover_2": "YouTube URL - Cover Song 2",
    "cover_3": "YouTube URL - Cover Song 3",
    "cover_4": "YouTube URL - Cover Song 4",
    "cover_5": "YouTube URL - Cover Song 5",
    "cover_6": "YouTube URL - Cover Song 6",
    "cover_7": "YouTube URL - Cover Song 7",
    "cover_8": "YouTube URL - Cover Song 8",
    "cover_9": "YouTube URL - Cover Song 9",
    "cover_10": "YouTube URL - Cover Song 10",
}


def concat_csvs(csv_paths, output_csv_path):
    
    def transform_url(url):
        url = url.split('&')[0] 
        url = url.split('=')[-1] if "=" in url else url.split("/")[-1]
        return url
    
    def common_member(a, b):
        a_set = set(
            [transform_url(link) for link in a]
        )
        b_set = set(
            [transform_url(link) for link in b]
        )
        if len(a_set.intersection(b_set)) > 0:
            return True
        return False
    
    def union(a,b):
        a_set = set(
            [transform_url(link) for link in a]
        )
        b_set = set(
            [transform_url(link) for link in b]
        )
        
        union = a_set.union(b_set)
        return ['https://youtu.be/' + l for l in union]
        

    output_links = pd.DataFrame({name: [] for name in COL_NAMES.values()})
    for path in csv_paths:
        new_dataset = pd.read_csv(path)
        to_be_added = []
        to_be_deleted_ids = []
        for idx_1, links_1 in new_dataset.iterrows():
            links_list_1 = links_1.dropna().values.tolist()
            new_links = links_list_1

            for idx_2, links_2 in output_links.iterrows():
                links_list_2 = links_2.dropna().values.tolist()
                
                if common_member(links_list_1[1:], links_list_2[1:]):
                    print(
                        f"Warning, duplicate song found: {links_1}, {links_2}. Will keep common elements..."
                    )
                    new_links = [links_list_1[0]] + union(links_list_1[1:], links_list_2[1:])
                    to_be_deleted_ids.append(idx_2)
                    break
                
            to_be_added.append(new_links)

        to_be_added = pd.DataFrame(to_be_added)
        to_be_added.columns = list(COL_NAMES.values())[:len(to_be_added.columns)]
        
        output_links = output_links.drop(to_be_deleted_ids)
        output_links = pd.concat([output_links, to_be_added])

    output_links.to_csv(output_csv_path)


def read_csv_and_download_songs(csv_path, output_base_path):
    downloader = YoutubeDownloader()
    links = pd.read_csv(csv_path)

    for index, row in tqdm(links.iterrows(), total=links.shape[0]):
        # Create folder
        song_name = row[COL_NAMES["song_name"]]
        song_base_path = output_base_path + "/" + song_name
        os.makedirs(song_base_path, exist_ok=True)

        # Download all covers to folder
        for cover in list(row.dropna())[1:]:
            downloader.download(song_base_path, cover)


def convert_songs_to_features(
    feature="hpcp",
    origin_path="/content/customdataset",
    save_path="/content/customdataset_hpcps",
):

    extractor = FeatureExtractor(feature)
    entries = os.listdir(origin_path)

    songs = {}
    if feature == "mfcc":
        feature = "XMFCC"
    elif feature == "hpcp":
        feature = "XHPCP"
    elif feature == "cens":
        feature = "XCENS"

    for dir in tqdm(entries):
        subdir = os.listdir(origin_path + "/" + dir)
        songs[dir] = []
        for song in subdir:
            song_id = dir
            cover_id = song.split(".mp3")[0]
            save_path_file = save_path + "/" + dir + "/" + cover_id + ".mat"
            if os.path.exists(save_path_file):
                print(f"Song: {song_id}/{cover_id} exists, skipping...")
                continue

            mat = {}
            mat[feature] = extractor.extract(origin_path + "/" + dir + "/" + song)
            os.makedirs(
                os.path.dirname(save_path + "/" + dir + "/" + song), exist_ok=True
            )
            savemat(save_path_file, mat)
            

def run_pipeline(csv_dir):
    
    def absoluteFilePaths(directory):
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
    
    csv_list = [path for path in absoluteFilePaths(csv_dir)]
    print(csv_list)
    
    output_csv = '/content/full_dataset.csv'
    output_dataset_songs = '/content/full_dataset_songs'
    concat_csvs(csv_list, output_csv)
    
    print("Downloading songs")
    read_csv_and_download_songs(output_csv, output_dataset_songs)
    
    print("Generating features")
    convert_songs_to_features(feature="hpcp", origin_path=output_dataset_songs, save_path="/content/full_dataset_hpcp")