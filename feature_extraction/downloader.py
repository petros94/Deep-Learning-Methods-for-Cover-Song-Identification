from ast import arg
import os 
import argparse

class YoutubeDownloader:        
    def download(self, path, link: str):
        os.chdir(path)
        os.system(f'yt-dlp -x --audio-format mp3 --audio-quality 0 {link}')
        
if __name__ == '__main__':
    curr_path = os.path.curdir
    try:
        os.mkdir(curr_path + "/tmp")
    except FileExistsError:
        pass
    
    yt = YoutubeDownloader()
    
    parser = argparse.ArgumentParser("Youtube downloader")
    parser.add_argument("-i", "--input")
    args = parser.parse_args()
    yt.download(curr_path + "/tmp", args.input)
    