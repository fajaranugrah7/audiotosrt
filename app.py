from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import whisper
import srt
from datetime import timedelta
import os

app = FastAPI()
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    result = model.transcribe(audio_path)
    subtitles = []
    for i, seg in enumerate(result["segments"]):
        subtitles.append(srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=seg["start"]),
            end=timedelta(seconds=seg["end"]),
            content=seg["text"].strip()
        ))

    srt_path = "output.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    os.remove(audio_path)
    return FileResponse(srt_path, filename="output.srt", media_type="text/plain")
