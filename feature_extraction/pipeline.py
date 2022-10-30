from feature_extraction.downloader import YoutubeDownloader
from feature_extraction.extraction import FeatureExtractor
import pandas as pd
import os
from scipy.io import savemat

COL_NAMES = {
    "song_name": "Song name",
    "original_song": "YouTube URL - Original Song",
    "cover_1": "YouTube URL - Cover Song 1",
    "cover_2": "YouTube URL - Cover Song 2",
    "cover_3": "YouTube URL - Cover Song 3",
}


def concat_csvs(csv_paths, output_csv_path):
    def common_member(a, b):
        a_set = set(
            [
                link.split("=")[-1] if "=" in link else link.split("/")[-1]
                for link in a
            ]
        )
        b_set = set(
            [
                link.split("=")[-1] if "=" in link else link.split("/")[-1]
                for link in b
            ]
        )
        if len(a_set.intersection(b_set)) > 0:
            return True
        return False

    output_links = pd.DataFrame()
    for path in csv_paths:
        new_dataset = pd.read_csv(path)
        to_be_added = []
        for links_1 in new_dataset.iterrows():
            skip = False

            for links_2 in output_links.iterrows():
                if common_member(links_1.tolist(), links_2.tolist()):
                    skip = True
                    print(
                        f"Warning, duplicate song found: {links_1}, {links_2}. Skipping..."
                    )
                    break

            if not skip:
                to_be_added.append(links_1)

        to_be_added = pd.DataFrame(to_be_added)
        output_links = pd.concat([output_links, to_be_added])

    output_links.to_csv(output_csv_path)


def read_csv_and_download_songs(csv_path, output_base_path):
    downloader = YoutubeDownloader()
    links = pd.read_csv(csv_path)

    for index, row in links.iterrows():
        # Create folder
        song_name = row[COL_NAMES["song_name"]]
        song_base_path = output_base_path + "/" + song_name
        os.makedirs(song_base_path, exist_ok=True)

        # Download all covers to folder
        for cover in list(row)[1:]:
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

    for dir in entries:
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
