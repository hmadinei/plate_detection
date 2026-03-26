import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline
from speechbrain.pretrained import EncoderClassifier
from sklearn.covariance import EllipticEnvelope
from tempfile import TemporaryDirectory
import pandas as pd
from typing import Optional
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from io import BytesIO

app = FastAPI()

from pathlib import Path


def fix_config_paths(config_file_path: str):
    config_path = Path(config_file_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    text = config_path.read_text(encoding='utf-8')
    fixed_text = text.replace('\\', '/')
    config_path.write_text(fixed_text, encoding='utf-8')
    print(f"Backslashes replaced in: {config_file_path}")


fix_config_paths("local_models/speaker-diarization-3.1/config.yaml")
fix_config_paths("local_models/spkrec-ecapa-voxceleb/hyperparams.yaml")

hf_token = "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"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("detected device : ", device)

# Define local directory
local_path = r"./local_models"

# Create subdirectories for segmentation and embedding models
segmentation_dir = os.path.join(local_path, "speaker-diarization-3.1/segmentation")
embedding_dir = os.path.join(local_path, "speaker-diarization-3.1/embedding")  # todo check embedding model for stage 2

# Set the torchaudio backend to FFMPEG to support m4a and other complex formats
# torchaudio.set_audio_backend("ffmpeg")

# os.makedirs(local_path, exist_ok=True)
# os.makedirs(segmentation_dir, exist_ok=True)
# os.makedirs(embedding_dir, exist_ok=True)


# # Step 1: Download pipeline config.yaml
# hf_hub_download(
#     repo_id="pyannote/speaker-diarization-3.1",
#     filename="config.yaml",
#     local_dir=local_path,
#     token=hf_token
# )

# # Step 2: Download segmentation model files (pyannote/segmentation-3.0)
# try:
#     hf_hub_download(
#         repo_id="pyannote/segmentation-3.0",
#         filename="config.yaml",
#         local_dir=segmentation_dir,
#         token=hf_token
#     )
#     hf_hub_download(
#         repo_id="pyannote/segmentation-3.0",
#         filename="pytorch_model.bin",
#         local_dir=segmentation_dir,
#         token=hf_token
#     )
# except Exception as e:
#     print(f"Error downloading segmentation model: {e}")
#     print("Ensure you have accepted the license for pyannote/segmentation-3.0 on Hugging Face.")
#     exit(1)
#
# # Step 3: Download embedding model files (pyannote/wespeaker-voxceleb-resnet34-LM)
# try:
#     hf_hub_download(
#         repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
#         filename="config.yaml",
#         local_dir=embedding_dir,
#         token=hf_token
#     )
#     hf_hub_download(
#         repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
#         filename="pytorch_model.bin",
#         local_dir=embedding_dir,
#         token=hf_token
#     )
# except Exception as e:
#     print(f"Error downloading embedding model: {e}")
#     print("Ensure you have accepted the license for pyannote/wespeaker-voxceleb-resnet34-LM on Hugging Face.")
#     exit(1)

# Step 4: Verify downloaded files
# print("Files in local directory:")
# print(os.listdir(local_path))
# print("Files in segmentation directory:")
# print(os.listdir(segmentation_dir))
# print("Files in embedding directory:")
# print(os.listdir(embedding_dir))

# try:
pipeline = Pipeline.from_pretrained("./local_models/speaker-diarization-3.1/config.yaml").to(device)
print("Pipeline loaded successfully!")
# except Exception as e:
#     print(f"Error loading pipeline: {e}")
#     exit(1)

# # Step 6: Test the pipeline
# audio_file = "path/to/your/audio.wav"  # Replace with your audio file path
# try:
#     diarization = pipeline(audio_file)
#     print("Diarization result:")
#     print(diarization)
# except Exception as e:
#     print(f"Error running diarization: {e}")


encoder_path = os.path.join(local_path, "spkrec-ecapa-voxceleb")

# ## second model:
# # Step 1: Download model files
# try:
#     # Download hyperparams.yaml
#     hf_hub_download(
#         repo_id="speechbrain/spkrec-ecapa-voxceleb",
#         filename="hyperparams.yaml",
#         local_dir=encoder_path,
#         token=hf_token
#     )
#     # Download embedding_model.ckpt
#     hf_hub_download(
#         repo_id="speechbrain/spkrec-ecapa-voxceleb",
#         filename="embedding_model.ckpt",
#         local_dir=encoder_path,
#         token=hf_token
#     )
#     # Download mean_var_norm_emb.ckpt
#     hf_hub_download(
#         repo_id="speechbrain/spkrec-ecapa-voxceleb",
#         filename="mean_var_norm_emb.ckpt",
#         local_dir=encoder_path,
#         token=hf_token
#     )
#     print("Model files downloaded successfully!")
# except Exception as e:
#     print(f"Error downloading model files: {e}")
#     exit(1)
#
# # Step 2: Verify downloaded files
# print("Files in local directory:")
# print(os.listdir(local_path))

# hf_hub_download(
#         repo_id="speechbrain/spkrec-ecapa-voxceleb",
#         filename="classifier.ckpt",
#         local_dir=encoder_path,
#         token=hf_token
#     )

try:
    embedding_model = EncoderClassifier.from_hparams(
        source=encoder_path,
        run_opts={"device": device}
    )
    print("Embedding model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


# Helper functions
def load_audio(filepath, target_sr=16000):
    waveform, sr = torchaudio.load(filepath)

    # print("in load audio waveform.shape = ",waveform.shape)

    # Downmix stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0), target_sr


def get_ecapa_embedding(waveform):
    if waveform.device != device:
        waveform = waveform.to(device)
    embedding = embedding_model.encode_batch(waveform.unsqueeze(0)).squeeze().detach()
    return embedding.cpu().numpy()


def remove_outliers(embeddings):
    if len(embeddings) < 2:
        return embeddings
    clf = EllipticEnvelope(contamination=0.1)
    try:
        clf.fit(embeddings)
        inliers = clf.predict(embeddings) == 1
        return embeddings[inliers]
    except Exception:
        # If error in fitting (e.g., very few points), return as is
        return embeddings


import subprocess
import tempfile
import torchaudio
import soundfile as sf
from fastapi.responses import StreamingResponse


def convert_to_wav(input_path, target_sr=16000):
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    # Use ffmpeg to decode into wav
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(target_sr), "-ac", "1",  # resample to 16kHz mono
        tmp_wav
    ], check=True)

    return tmp_wav


# API Endpoints
@app.post("/signature")
async def generate_signature(audio: UploadFile = File(...), get_embedding_file: bool = Form(False)):
    with TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, audio.filename)
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        # Diarization
        audio_path = convert_to_wav(audio_path)
        diarization = pipeline(audio_path)
        diarization_segments = []

        for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                "start": speech_turn.start,
                "end": speech_turn.end,
                "speaker": speaker
            })

        print("diarization_segments = ", diarization_segments)

        diarization_df = pd.DataFrame(diarization_segments)

        # Load original audio
        waveform, sr = load_audio(audio_path)

        speaker_embeddings = {}

        for speaker in diarization_df["speaker"].unique():
            speaker_df = diarization_df[diarization_df["speaker"] == speaker]
            embeddings = []

            for _, row in speaker_df.iterrows():
                start_sample = int(row["start"] * sr)
                end_sample = int(row["end"] * sr)
                segment = waveform[start_sample:end_sample]

                # Skip very short segments
                if segment.shape[0] < int(0.16 * sr):  # less than 160ms
                    continue

                segment_waveform = segment.squeeze(0)
                # print("segment_waveform.shape = " ,segment_waveform.shape)
                emb = get_ecapa_embedding(segment_waveform)
                embeddings.append(emb)

            embeddings = np.array(embeddings)
            embeddings = remove_outliers(embeddings)

            if len(embeddings) > 0:
                mean_embedding = np.mean(embeddings, axis=0)
                # save_path =  f"speaker_{speaker}.npy"
                # np.save(save_path, mean_embedding)
                speaker_embeddings[speaker] = {
                    "embedding": mean_embedding.tolist(),
                    "segments": [
                        {"start": row["start"], "end": row["end"]}
                        for _, row in speaker_df.iterrows()
                    ]
                }

                if get_embedding_file:
                    # Example embedding (replace with your actual one)
                    # Save as npy in-memory
                    buffer = BytesIO()
                    np.save(buffer, mean_embedding)
                    buffer.seek(0)

                    # Return as downloadable npy file
                    return StreamingResponse(
                        buffer,
                        media_type="application/octet-stream",
                        headers={
                            "Content-Disposition": "attachment; filename=embedding.npy"
                        }
                    )

        return {"result": speaker_embeddings}


@app.post("/match")
async def match_signature(signature_file: UploadFile = File(...), audio_file: UploadFile = File(...),
                          threshold: Optional[float] = Query(0.7, ge=0.0, le=1.0)):
    with TemporaryDirectory() as tmpdir:
        # Save uploaded signature_vector npy file
        signature_path = os.path.join(tmpdir, signature_file.filename)
        with open(signature_path, "wb") as f:
            f.write(await signature_file.read())

        try:
            signature_vector = np.load(signature_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Error loading signature npy: {str(e)}")

        # Save uploaded audio file
        audio_path = os.path.join(tmpdir, audio_file.filename)
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

        # --- Diarization ---
        audio_path = convert_to_wav(audio_path)
        diarization = pipeline(audio_path)
        diarization_segments = []

        for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                "start": speech_turn.start,
                "end": speech_turn.end,
                "speaker": speaker
            })

        diarization_df = pd.DataFrame(diarization_segments)

        # --- Load waveform ---
        waveform, sr = load_audio(audio_path)

        results = {}

        for speaker in diarization_df["speaker"].unique():
            speaker_df = diarization_df[diarization_df["speaker"] == speaker]
            embeddings = []

            for _, row in speaker_df.iterrows():
                start_sample = int(row["start"] * sr)
                end_sample = int(row["end"] * sr)
                segment = waveform[start_sample:end_sample]

                if segment.shape[0] < int(0.16 * sr):  # skip very short segments
                    continue

                emb = get_ecapa_embedding(segment)
                embeddings.append(emb)

            embeddings = np.array(embeddings)
            embeddings = remove_outliers(embeddings)

            if len(embeddings) > 0:
                mean_embedding = np.mean(embeddings, axis=0)
                # Cosine similarity
                score = np.dot(mean_embedding, signature_vector) / (
                        np.linalg.norm(mean_embedding) * np.linalg.norm(signature_vector)
                )
                match = score > threshold  # Threshold

                total_speaking_time = sum(row["end"] - row["start"] for _, row in speaker_df.iterrows())

                results[speaker] = {
                    "match": bool(match),
                    "score": round(float(score), 4),
                    "segments": [
                        {"start": round(row["start"], 2), "end": round(row["end"], 2)}
                        for _, row in speaker_df.iterrows()
                    ],
                    "total_time_seconds": round(total_speaking_time, 2)
                }

        return {"result": results}

    # with TemporaryDirectory() as tmpdir:
    #     signature_path = os.path.join(tmpdir, signature_file.filename)
    #     with open(signature_path, "wb") as f:
    #         f.write(await signature_file.read())
    #
    #     # Load signature vector from .npy file
    #     try:
    #         signature_vector = np.load(signature_path)
    #         if not isinstance(signature_vector, np.ndarray) or signature_vector.ndim != 1:
    #             raise ValueError("Signature file must be a 1D numpy array.")
    #     except Exception as e:
    #         raise HTTPException(status_code=422, detail=f"Invalid signature file: {str(e)}")
    #
    #
    #     audio_path = os.path.join(tmpdir, audio_file.filename)
    #     with open(audio_path, "wb") as f:
    #         f.write(await audio_file.read())
    #
    #         # Load full audio
    #         waveform, sr = load_audio(audio_path)  # ⚡ Make sure you have a `load_audio` function
    #
    #         # Get the whole audio embedding
    #         if waveform.dim() == 1:
    #             print("something is suspicious, waveform.dim() must be 1 but it is = ",waveform.dim())
    #             waveform = waveform.unsqueeze(0)  # Make sure it is (1, num_samples)
    #             print(waveform.dim())
    #
    #         # Remove very short audios
    #         if waveform.shape[1] < int(0.16 * sr):  # less than 160 ms
    #             raise HTTPException(status_code=400, detail="Audio is too short for meaningful comparison.")
    #
    #         # Get the ECAPA embedding
    #         segment_waveform = waveform.squeeze(0)
    #         embedding = get_ecapa_embedding(segment_waveform)
    #
    #         # Calculate cosine similarity
    #         embedding = np.array(embedding)
    #         signature_vector = np.array(signature_vector)
    #         score = np.dot(embedding, signature_vector) / (np.linalg.norm(embedding) * np.linalg.norm(signature_vector))
    #
    #         match = score > 0.7  # you can adjust threshold
    #
    #         return JSONResponse(content={"match": bool(match), "score": round(float(score), 4)})


