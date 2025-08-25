"""
sudo apt install aria2 p7zip-full ffmpeg -y
aria2c -x16 https://huggingface.co/datasets/thewh1teagle/saspeech/resolve/main/saspeech_automatic/saspeech_automatic.7z -o saspeech_automatic.7z
7z x saspeech_automatic.7z
"""

import stable_whisper
from pathlib import Path
from tqdm import tqdm
import json

src_folder = Path('saspeech_automatic/wav')
dst_folder = Path('saspeech_automatic/transcripts')
dst_folder.mkdir(parents=True, exist_ok=True)

def main():
    model = stable_whisper.load_faster_whisper('ivrit-ai/whisper-large-v3-turbo-ct2')
    files = list(src_folder.glob('*.wav'))
    for audio_file in tqdm(files):
        target_file = dst_folder / f'{audio_file.stem}.json'
        segs = model.transcribe(str(audio_file), language='he', word_timestamps=True) # Word level timestamps enabled by default
        words = segs.all_words()
        segs_dict = {
            'text': segs.text,
            'segments': [
                {
                    'start': w.start,
                    'end': w.end,
                    'text': w.word
                } for w in words
            ]
        }
        with open(target_file, 'w') as f:
            json.dump(segs_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()