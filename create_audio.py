import json
from pathlib import Path
from tqdm import tqdm
import pydub

# פרמטרים
MIN_DUR = 1.0      # מינימום אורך סגמנט (שניות)
MAX_DUR = 5.0      # מקסימום אורך סגמנט (שניות)
GAP_REQUIRED = 0.3 # פער מינימום לסגמנט הבא
SILENCE_DUR = 0.3  # padding אמיתי מהאודיו לפני ואחרי
SPEED_TOL = 0.2    # טולרנס
STRICT_SLOW = True # אם True → מסנן החוצה מהירים מדי, משאיר רק איטיים

src_folder = Path("saspeech_automatic")
dst_folder = Path("saspeech_automatic_short")
dst_folder.mkdir(parents=True, exist_ok=True)

def load_transcript(transcript_file):
    with open(transcript_file, "r") as f:
        return json.load(f)

def calc_speech_rates(transcript_files):
    rates = []
    for transcript_file in tqdm(transcript_files, desc="Calculating speech rates"):
        transcript = load_transcript(transcript_file)
        for seg in transcript["segments"]:
            duration = seg["end"] - seg["start"]
            n_words = len(seg["text"].split())
            if duration > 0 and n_words > 0:
                rates.append(n_words / duration)
    if not rates:
        return 0
    return sum(rates) / len(rates)

def save_chunk(audio, buffer, start_time, end_time, base_name, out_audio_folder, out_transcript_folder):
    real_start = max(0, (start_time - SILENCE_DUR) * 1000)
    real_end = min(len(audio), int(end_time * 1000) + int(SILENCE_DUR * 1000))
    chunk_audio = audio[real_start:real_end]

    out_wav = f"{base_name}.wav"
    out_path = out_audio_folder / out_wav
    chunk_audio.export(out_path, format="wav")

    out_json = {
        "text": " ".join([b["text"].strip() for b in buffer]),
        "segments": buffer,
        "start": (start_time - SILENCE_DUR) if start_time - SILENCE_DUR > 0 else 0,
        "end": end_time + SILENCE_DUR
    }
    out_json_file = out_transcript_folder / f"{base_name}.json"
    with open(out_json_file, "w") as f_out:
        json.dump(out_json, f_out, ensure_ascii=False, indent=2)

    return len(chunk_audio) / 1000.0  # מחזיר אורך שניות

def process_file(transcript_file, audio_folder, out_audio_folder, out_transcript_folder, mean_rate):
    base_name = transcript_file.stem
    audio_file = audio_folder / f"{base_name}.wav"
    audio = pydub.AudioSegment.from_wav(audio_file)

    transcript = load_transcript(transcript_file)
    segments = transcript["segments"]

    buffer = []
    start_time = None

    if STRICT_SLOW:
        THRESHOLD = mean_rate * (1 - SPEED_TOL)
    else:
        LOWER = mean_rate * (1 - SPEED_TOL)
        UPPER = mean_rate * (1 + SPEED_TOL)

    for i, seg in enumerate(segments):
        if start_time is None:
            start_time = seg["start"]

        buffer.append(seg)
        end_time = buffer[-1]["end"]
        duration = end_time - start_time

        if seg["text"].strip().endswith((".", ",", "!", "?")):
            if MIN_DUR <= duration <= MAX_DUR:
                n_words = len(" ".join([b["text"] for b in buffer]).split())
                rate = n_words / duration if duration > 0 else 0

                # בדיקת מהירות
                if STRICT_SLOW:
                    if rate > THRESHOLD:
                        continue
                else:
                    if not (LOWER <= rate <= UPPER):
                        continue

                if i + 1 < len(segments):
                    next_start = segments[i + 1]["start"]
                    gap = next_start - end_time
                    if gap < GAP_REQUIRED:
                        continue

                # שומרים ומחזירים אורך
                return save_chunk(audio, buffer, start_time, end_time, base_name,
                                  out_audio_folder, out_transcript_folder)

    return 0.0  # לא נשמר כלום

def main():
    transcript_folder = src_folder / "transcripts"
    audio_folder = src_folder / "wav"
    out_audio_folder = dst_folder / "wav"
    out_transcript_folder = dst_folder / "transcripts"

    out_audio_folder.mkdir(parents=True, exist_ok=True)
    out_transcript_folder.mkdir(parents=True, exist_ok=True)

    transcript_files = list(transcript_folder.glob("*.json"))

    mean_rate = calc_speech_rates(transcript_files)
    print(f"Mean speech rate: {mean_rate:.2f} words/sec")

    total_duration = 0.0
    kept = 0
    skipped = 0

    for transcript_file in tqdm(transcript_files, desc="Processing transcripts"):
        dur = process_file(transcript_file, audio_folder, out_audio_folder, out_transcript_folder, mean_rate)
        if dur > 0:
            total_duration += dur
            kept += 1
        else:
            skipped += 1

    h = int(total_duration // 3600)
    m = int((total_duration % 3600) // 60)
    s = int(total_duration % 60)
    print(f"\nKept {kept} files, skipped {skipped}")
    print(f"Total duration in new folder: {h:02d}:{m:02d}:{s:02d}")

if __name__ == "__main__":
    main()
