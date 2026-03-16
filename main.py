import os
import uuid
import subprocess
import tempfile
import shutil
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://moneyafrica-clipper.netlify.app",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = tempfile.gettempdir()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def get_ffmpeg():
    for path in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        found = shutil.which(path)
        if found:
            return found
    return None


@app.get("/")
def root():
    ffmpeg = get_ffmpeg()
    return {"status": "Clip Agent backend is running", "ffmpeg": ffmpeg or "not found"}


@app.get("/health")
def health():
    ffmpeg = get_ffmpeg()
    return {"ffmpeg_path": ffmpeg, "ffmpeg_found": ffmpeg is not None}


class ClaudeRequest(BaseModel):
    prompt: str
    max_tokens: int = 1000


@app.post("/claude")
async def claude_proxy(req: ClaudeRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set on server")

    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": req.max_tokens,
                "messages": [{"role": "user", "content": req.prompt}],
            },
        )

    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail=res.text)

    return res.json()


@app.post("/cut-clip")
async def cut_clip(
    video: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...),
    clip_title: str = Form(default="clip"),
):
    ffmpeg_path = get_ffmpeg()
    if not ffmpeg_path:
        raise HTTPException(status_code=500, detail="FFmpeg not found on server")

    input_id = uuid.uuid4().hex
    ext = os.path.splitext(video.filename)[-1] or ".mp4"
    input_path = os.path.join(TEMP_DIR, f"input_{input_id}{ext}")
    output_path = os.path.join(TEMP_DIR, f"clip_{input_id}.mp4")

    try:
        with open(input_path, "wb") as f:
            content = await video.read()
            f.write(content)

        cmd = [
            ffmpeg_path, "-y",
            "-i", input_path,
            "-ss", str(start),
            "-to", str(end),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr[-500:]}")

        safe_title = "".join(c for c in clip_title if c.isalnum() or c in " _-").strip().replace(" ", "_")
        filename = f"{safe_title or 'clip'}.mp4"

        return FileResponse(output_path, media_type="video/mp4", filename=filename)

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


if __name__ == "__main__":
    import uvicorn
    # Strip any whitespace and convert to int safely
    raw_port = os.environ.get("PORT", "8000").strip()
    port = int(raw_port)
    uvicorn.run(app, host="0.0.0.0", port=port)
