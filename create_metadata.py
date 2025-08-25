"""
pip install -U phonikud-onnx phonikud

wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
uv run examples/with_phonikud_onnx.py
"""

from pathlib import Path
import json
from tqdm import tqdm
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

from phonikud_onnx import Phonikud
from phonikud import phonemize as phonemize_fn


# --- Initialize ONNX session with optimizations ---
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "./phonikud-1.0.int8.onnx",
    sess_options=so,
    # We have no time to wait, so we use GPU if available
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
model = Phonikud.from_session(session)


def phonemize(text: str):
    with_diacritics = model.add_diacritics(text)
    phonemes = phonemize_fn(with_diacritics)
    return with_diacritics, phonemes


def phonemize_record(transcript_file: Path):
    """Load transcript JSON, run diacritics + phonemize, return sanitized result"""
    with transcript_file.open("r", encoding="utf-8") as f:
        transcript = json.load(f)

    text = transcript.get("text", "")
    text_with_diac, phones = phonemize(text)

    # sanitize tabs/newlines
    text_with_diac = text_with_diac.replace("\t", " ").replace("\n", " ")
    phones = phones.replace("\t", " ").replace("\n", " ")

    return transcript_file.stem, text_with_diac, phones


def main():
    src_folder = Path("saspeech_automatic_short")
    transcript_folder = src_folder / "transcripts_saspeech_automatic_short"
    transcript_files = sorted(transcript_folder.glob("*.json"), key=lambda x: int(x.stem))
    # transcript_files = transcript_files[:200]
    dst_file = src_folder / "metadata.csv"

    with dst_file.open("w", encoding="utf-8", newline="") as out_f:
        # Run phonemize_record in parallel, order preserved
        # We have no time to wait
        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(
                tqdm(
                    ex.map(phonemize_record, transcript_files),
                    total=len(transcript_files),
                    desc="Processing transcripts",
                )
            )

        # Write in the same order
        for stem, text_with_diac, phones in results:
            out_f.write(f"{stem}\t{text_with_diac}\t{phones}\n")


if __name__ == "__main__":
    main()
