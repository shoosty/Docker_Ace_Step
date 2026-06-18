"""ACE-Step v60 — 1.5 XL handler with sweet-spot recipe + mastering.

v60 (Stephen 2026-06-18, later same day): adds opt-in two-pass loudnorm
     mastering. Set master=true on the input to enable. The handler runs
     ffmpeg's loudnorm filter (EBU R128 spec) before the MP3 transcode,
     so audio_url comes back at -14 LUFS / -1 dBTP. When mastering is on
     the MP3 bitrate bumps from 192 to 320 kbps so the codec doesn't
     undo the polish. The raw unmastered WAV is preserved when
     keep_wav=true — that's the artist's "I want to do my own mastering"
     escape hatch. Idempotent in loudness (already-mastered input parses
     as already at -14 and the second pass corrects ~nothing) but
     quality-degrading on MP3->MP3 chains; the caller should track a
     mastered_at flag and refuse to re-master the same audio_url.

     Cost: ~9 seconds of CPU on a 4-minute song. GPU is idle during
     the mastering pass, so it's effectively free time.

     Still unfixed (deferred to v61 with image-side surgery): flash_attn
     CUDA-12.8 ABI rebuild + nano-vllm triton pin.

v59 (Stephen 2026-06-18): v58's DCW-on + guidance-7 defaults made
     the model sound terrible — confirmed across a 9-test E-series
     listening matrix (E1=DCW-off was clean; E3=DCW-double was the
     "terrible" baseline; E7=guidance-10 was the loved single-knob
     bump). Two interactions worth knowing: (a) combining 50 steps
     with guidance 10 (E10) broke it — knobs don't compound, pick
     one bump or the other, and (b) raising guidance past 10 (E12)
     muffled the vocal. v59 flips just the two defaults that the
     E-series proved correct: dcw_enabled=False, guidance_scale=10.0.
     Steps stays at 28; ODE sampler; shift 3.0; cfg window 0.0->0.95.
     Per-request callers can still override anything — the studio
     stopped sending overrides as of the same date.

     Known unsolved (NOT addressed in v59): flash_attn fails to
     import on RTX 5090 + CUDA 12.8 (symbol mismatch); nano-vllm
     fails on the same hardware due to a triton ABI break
     (triton_key was removed). Both fall back to PyTorch — works
     but slower + more CPU-bound. Real fix is a CUDA-12.8 flash_attn
     rebuild + triton pin in a future v60.

v58 (Stephen 2026-06-16): pass full quality params to GenerationParams.
     Previously passing only caption/duration/lyrics — model was running
     on internal defaults causing bad audio. Now passes inference_steps,
     guidance_scale, shift, infer_method, cfg_interval bounds, bpm,
     keyscale, vocal_language, seed. Also activates DCW wavelet quality
     filter (dcw_enabled=True, mode="double") — packages were installed
     in v56 but never used. No Dockerfile changes — handler only.
     (Quality regression — see v59.)

v57 (Stephen 2026-06-13): the 12-minute song test reached end-of-
generation cleanly but failed when uploading the ~120MB WAV to
Supabase Storage — Storage rejects single objects over its
per-object cap with HTTP 413 ("Payload too large"). MP3 was already
uploaded successfully as the primary. v57 makes the keep_wav path
size-aware: if the WAV exceeds SUPABASE_MAX_OBJECT_BYTES (default
~49 MB to stay under the standard 50 MB cap with some headroom),
we skip the WAV upload, log a warning, and return the response
with `wav_skipped: true` so the caller can render the song from
the MP3 alone. We also catch upload-time exceptions and tag them
the same way — a 413 from the API still gracefully degrades."""
import runpod
import sys
import os
import base64
import tempfile
import traceback
import subprocess
import shutil
import time
import uuid
import urllib.request
import glob

sys.path.insert(0, '/ace-step-code')

MODEL_SIZE = os.environ.get("MODEL_SIZE", "xl").lower()

CHECKPOINTS_DIR = "/runpod-volume/checkpoints"
if not os.path.exists(CHECKPOINTS_DIR):
    raise RuntimeError(f"Checkpoints not found at {CHECKPOINTS_DIR}")

dit_variant = "acestep-v15-xl-base" if MODEL_SIZE == "xl" else "acestep-v15-turbo"
lm_variant = "acestep-5Hz-lm-1.7B"

os.environ["ACESTEP_CHECKPOINTS_DIR"] = CHECKPOINTS_DIR

try:
    print(f"Loading ACE-Step 1.5 (DiT={dit_variant}, LM={lm_variant})...")
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    init_status, enable_generate = dit_handler.initialize_service(
        project_root="/runpod-volume",
        config_path=dit_variant,
        device="cuda",
    )
    print(f"DEBUG dit init_status={init_status} enable_generate={enable_generate}")
    if not enable_generate:
        raise RuntimeError(f"DiT model initialization failed: {init_status}")

    llm_handler.initialize(
        checkpoint_dir=CHECKPOINTS_DIR,
        lm_model_path=lm_variant,
        backend="vllm",
        device="cuda",
    )
    print("Pipeline loaded!")
except Exception as e:
    import traceback
    print(f"STARTUP ERROR: {e}")
    print(traceback.format_exc())
    raise

if not shutil.which("ffmpeg"):
    print("WARNING: ffmpeg not on PATH — MP3 conversion will fall back to WAV.")

# ── Supabase client ────────────────────────────────────────────────
# Stephen 2026-06-12: accept either env name so a key-format rotation
# (legacy JWT eyJ... → newer sb_secret_...) doesn't break the upload.
# SUPABASE_SERVICE_ROLE_KEY is the legacy service-role JWT; the newer
# SUPABASE_SECRET_KEY is Supabase's "secret API key" format. Either
# is valid against Supabase Storage with full bucket permissions.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY_RAW = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_SECRET_KEY_RAW = os.environ.get("SUPABASE_SECRET_KEY")
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY_RAW or SUPABASE_SECRET_KEY_RAW
ACESTEP_BUCKET = os.environ.get("ACESTEP_BUCKET", "song-uploads")

# v57 — Supabase Storage rejects single objects over its per-object
# cap with HTTP 413 ("Payload too large"). The standard cap is 50 MB;
# we default to 49 MB to leave a sliver of multipart-overhead room.
# Operators can raise this with the env var if they've bumped their
# Storage settings.
SUPABASE_MAX_OBJECT_BYTES = int(
    os.environ.get("SUPABASE_MAX_OBJECT_BYTES", str(49 * 1024 * 1024))
)

# v54 diagnostic — Stephen 2026-06-13: the "RLS 403" we keep seeing
# from RunPod is almost certainly a missing/malformed env var, not a
# Supabase storage policy. Print exactly what env state the worker
# sees at container startup so the worker log tells the truth:
#   - which keys are present
#   - their lengths (a clean service-role JWT is ~221 chars; a paste
#     with embedded newlines / chat fragments will be longer or be
#     visibly noisy)
#   - first 6 + last 6 chars (not enough to leak the secret, enough
#     to confirm it starts with eyJ and ends with the right tail)
def _diag_keyfrag(name: str, value: str | None) -> str:
    if value is None:
        return f"{name}=<UNSET>"
    # Strip surrounding whitespace for the length report so we'd see
    # the difference between a clean 221-char JWT and a multi-line
    # paste of the same key.
    clean_len = len(value)
    stripped_len = len(value.strip())
    has_newline = "\n" in value or "\r" in value
    head = value[:6] if clean_len >= 6 else value
    tail = value[-6:] if clean_len >= 6 else ""
    extras = ""
    if has_newline:
        extras += " HAS_NEWLINES!"
    if stripped_len != clean_len:
        extras += f" trailing_ws={clean_len - stripped_len}"
    return f"{name} len={clean_len} head='{head}' tail='{tail}'{extras}"

print("[v54 env diag]")
print(f"  SUPABASE_URL={SUPABASE_URL!r}")
print(f"  {_diag_keyfrag('SUPABASE_SERVICE_ROLE_KEY', SUPABASE_SERVICE_ROLE_KEY_RAW)}")
print(f"  {_diag_keyfrag('SUPABASE_SECRET_KEY', SUPABASE_SECRET_KEY_RAW)}")
print(f"  ACESTEP_BUCKET={ACESTEP_BUCKET!r}")
if SUPABASE_KEY is None:
    print("  → NO Supabase key visible. Upload will fail with the same error.")
elif "\n" in (SUPABASE_KEY or ""):
    print("  → key contains newline(s). Re-paste cleanly in RunPod's env field.")
else:
    print(f"  → using key of length {len(SUPABASE_KEY)} for upload.")

_supabase_client = None
def supabase_client():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL plus either SUPABASE_SERVICE_ROLE_KEY (legacy JWT) "
            "or SUPABASE_SECRET_KEY (newer format) must be set on this "
            "RunPod endpoint for URL-based uploads."
        )
    from supabase import create_client
    _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client

def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
         "-codec:a", "libmp3lame", "-b:a", bitrate, mp3_path],
        check=True,
    )

def master_audio(input_wav_path: str, output_wav_path: str) -> dict:
    """v60 mastering pass — ffmpeg two-pass loudnorm.

    Pass 1 measures the input loudness (EBU R128). Pass 2 normalizes
    to streaming spec: I=-14 LUFS, LRA=11, true peak <= -1 dBTP.
    `linear=true` keeps the gain transparent (constant-curve EQ-free
    correction rather than the dynamic mode, which would re-compress).

    Returns the measured input stats so the caller can log them and
    decide whether to surface "already pretty hot" hints. Idempotent
    in loudness terms — running on an already-mastered file changes
    almost nothing because pass-1 reports the file is already at -14
    LUFS and pass-2 has nothing to correct."""
    import json as _json
    # Pass 1 — measure.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", input_wav_path,
         "-af", "loudnorm=I=-14:LRA=11:TP=-1:print_format=json",
         "-f", "null", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    # The JSON block is on stderr at the END of the run; find the
    # last {...} that parses cleanly.
    stderr = proc.stderr
    last_brace = stderr.rfind("}")
    start = stderr.rfind("{", 0, last_brace)
    measured = _json.loads(stderr[start:last_brace + 1])
    # Pass 2 — normalize using the measured stats.
    filter_chain = (
        "loudnorm=I=-14:LRA=11:TP=-1"
        f":measured_I={measured['input_i']}"
        f":measured_LRA={measured['input_lra']}"
        f":measured_TP={measured['input_tp']}"
        f":measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}"
        ":linear=true"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", input_wav_path,
         "-af", filter_chain,
         "-ar", "48000", "-ac", "2",
         "-codec:a", "pcm_f32le",
         output_wav_path],
        check=True,
    )
    return measured

def upload_to_supabase(local_path: str, storage_path: str, content_type: str) -> str:
    # v55 — reprint the env diagnostic right before the upload so the
    # diag block sits adjacent to the failure traceback (no scrolling
    # back to container startup logs). Stephen 2026-06-13.
    print(
        f"[v55 upload diag] bucket={ACESTEP_BUCKET!r} path={storage_path!r} "
        f"ct={content_type!r} file_bytes={os.path.getsize(local_path)}"
    )
    print(f"[v55 env diag (per-job)]")
    print(f"  SUPABASE_URL={SUPABASE_URL!r}")
    print(f"  {_diag_keyfrag('SUPABASE_SERVICE_ROLE_KEY', SUPABASE_SERVICE_ROLE_KEY_RAW)}")
    print(f"  {_diag_keyfrag('SUPABASE_SECRET_KEY', SUPABASE_SECRET_KEY_RAW)}")
    if SUPABASE_KEY is None:
        print("  → NO key visible. Upload WILL fail (was about to call Storage API).")
    elif "\n" in (SUPABASE_KEY or ""):
        print("  → key contains newline(s). HTTP header rejected → 400 Bad Request.")
    else:
        print(f"  → key length {len(SUPABASE_KEY)} (clean service-role JWT is 221).")
    sb = supabase_client()
    with open(local_path, "rb") as f:
        data = f.read()
    sb.storage.from_(ACESTEP_BUCKET).upload(
        path=storage_path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    res = sb.storage.from_(ACESTEP_BUCKET).get_public_url(storage_path)
    if isinstance(res, dict):
        return res.get("publicUrl") or res.get("publicURL") or res.get("public_url")
    return res

def download_to_temp(url: str, suffix: str = ".bin") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        local_path = f.name
    urllib.request.urlretrieve(url, local_path)
    return local_path

def find_audio_file(result, save_dir):
    """Locate the audio file from a GenerationResult."""
    if hasattr(result, "audios") and result.audios:
        first = result.audios[0]
        print(f"DEBUG audios[0] type: {type(first)}, value: {first}")
        if isinstance(first, str):
            return first
        if hasattr(first, "path"):
            return first.path
        if hasattr(first, "audio_path"):
            return first.audio_path
        if hasattr(first, "save_path"):
            return first.save_path
        if isinstance(first, dict):
            print(f"DEBUG dict keys: {first.keys()}")
            return first.get("path") or first.get("audio_path") or first.get("save_path")
        print(f"DEBUG audios[0] attrs: {dir(first)}")
        for attr in dir(first):
            if not attr.startswith("_"):
                val = getattr(first, attr)
                if isinstance(val, str) and val.endswith((".wav", ".mp3", ".flac")):
                    return val
    wavs = glob.glob(f"{save_dir}/*.wav") + glob.glob(f"{save_dir}/*.flac")
    if wavs:
        return wavs[0]
    return None

def handler(job):
    """RunPod serverless entrypoint."""
    src_temp = None
    lora_temp = None
    try:
        inp = job.get("input", {}) or {}

        caption  = inp.get("caption", "pop music")
        lyrics   = inp.get("lyrics",  "[Instrumental]")
        duration = float(inp.get("audio_duration", inp.get("duration", 30)))

        fmt = (inp.get("format") or "mp3").lower()
        if fmt not in ("mp3", "wav"):
            return {"error": f"format must be 'mp3' or 'wav', got '{fmt}'"}
        keep_wav   = bool(inp.get("keep_wav", False)) and fmt == "mp3"
        return_b64 = bool(inp.get("return_audio_b64", False))
        # v60 — opt-in mastering pass. When true the handler runs
        # ffmpeg loudnorm two-pass on the rendered WAV before any
        # MP3 transcode + upload. Streaming spec: I=-14 LUFS, true
        # peak <= -1 dBTP. When mastering is on, the MP3 bitrate
        # bumps from 192 to 320 kbps so the mastered output isn't
        # immediately hobbled by codec loss. Cost ~9s of CPU on a
        # 4-min song. Idempotent in loudness terms (already-mastered
        # input parses as already at -14 LUFS and pass-2 makes
        # near-zero correction).
        master     = bool(inp.get("master", False))

        ts       = int(time.time())
        short_id = uuid.uuid4().hex[:12]
        storage_path = inp.get("storage_path") or f"acestep-runs/{ts}-{short_id}.{fmt}"
        if not storage_path.endswith(f".{fmt}"):
            base, _, _ = storage_path.rpartition(".")
            storage_path = f"{base or storage_path}.{fmt}"
        storage_path_wav = inp.get("storage_path_wav")
        if keep_wav and not storage_path_wav:
            base, _, _ = storage_path.rpartition(".")
            storage_path_wav = f"{base}-wav.wav"

        lora_url = inp.get("lora_url")
        if lora_url:
            lora_temp = download_to_temp(lora_url, suffix=".safetensors")

        # ── per-request quality overrides ────────────────────────────────
        # Defaults are the v59 sweet-spot recipe (Stephen 2026-06-18) —
        # see module docstring for the E1-E12 listening series that
        # produced them. Callers can override any single value, BUT do
        # not combine inference_steps=50 with guidance_scale=10 — the
        # E10 test proved the two best individual knobs don't compound.
        inference_steps = int(inp.get("inference_steps",      28))
        guidance_scale  = float(inp.get("guidance_scale",     10.0))
        shift           = float(inp.get("shift",              3.0))
        infer_method    = inp.get("infer_method",             "ode")
        cfg_start       = float(inp.get("cfg_interval_start", 0.0))
        cfg_end         = float(inp.get("cfg_interval_end",   0.95))
        bpm             = inp.get("bpm",                      None)
        keyscale        = inp.get("keyscale",                 "")
        vocal_language  = inp.get("vocal_language",           "en")
        seed            = int(inp.get("seed",                 -1))

        # DCW wavelet quality filter — packages installed since v56.
        # v59 flipped the default to OFF after the E3 listening test
        # showed dcw_mode="double" produces "terrible" scrambled audio.
        # Valid dcw_mode values are ('low', 'high', 'double', 'pix');
        # 'single' is NOT valid (would raise from DCWCorrector). When
        # enabled, the DCW correction lives at the wavelet decompose
        # step and adds detail at higher frequencies — useful for
        # specific styles, but the default-off recipe is what users
        # actually want.
        dcw_enabled     = bool(inp.get("dcw_enabled",         False))
        dcw_mode        = inp.get("dcw_mode",                 "double")
        dcw_scaler      = float(inp.get("dcw_scaler",         0.05))
        dcw_high_scaler = float(inp.get("dcw_high_scaler",    0.02))

        params = GenerationParams(
            caption=caption,
            duration=duration,
            lyrics=lyrics if lyrics else "",
            task_type=inp.get("task_type", "text2music"),
            vocal_language=vocal_language,
            bpm=bpm,
            keyscale=keyscale,
            seed=seed,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            infer_method=infer_method,
            cfg_interval_start=cfg_start,
            cfg_interval_end=cfg_end,
            dcw_enabled=dcw_enabled,
            dcw_mode=dcw_mode,
            dcw_scaler=dcw_scaler,
            dcw_high_scaler=dcw_high_scaler,
        )
        if lora_temp:
            params.lora_path = lora_temp
            params.lora_weight = float(inp.get("lora_weight", 1.0))

        config = GenerationConfig(batch_size=1, audio_format="wav")

        with tempfile.TemporaryDirectory() as save_dir:
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=save_dir)

            print(f"DEBUG success={result.success} error={result.error} status={result.status_message} audios={result.audios}")

            if not result.success:
                return {"error": f"generate_music failed: {result.error} | {result.status_message}"}

            wav_path = find_audio_file(result, save_dir)
            if not wav_path:
                return {"error": f"No audio file found. Result attrs: {dir(result)}"}

            # v60 — opt-in mastering. Runs BEFORE the MP3 transcode so
            # the mp3 carries the mastered audio (and at the bumped
            # bitrate so we don't undo the polish with codec loss).
            # The raw unmastered WAV is preserved at wav_path so it's
            # still what gets uploaded if keep_wav is requested — the
            # idea is that the artist always has the raw render available
            # for their own offline mastering workflow.
            mastered_stats = None
            if master:
                mastered_wav = os.path.join(save_dir, "mastered.wav")
                print(f"[v60 master] running loudnorm two-pass on {wav_path}")
                mastered_stats = master_audio(wav_path, mastered_wav)
                print(
                    f"[v60 master] measured I={mastered_stats.get('input_i')} "
                    f"LRA={mastered_stats.get('input_lra')} "
                    f"TP={mastered_stats.get('input_tp')} "
                    f"offset={mastered_stats.get('target_offset')}"
                )
                # Replace the working wav for downstream transcode/upload.
                # Raw render at wav_path stays untouched for keep_wav.
                rendered_wav = wav_path  # remember for keep_wav branch
                wav_for_primary = mastered_wav
            else:
                rendered_wav = wav_path
                wav_for_primary = wav_path

            mp3_path = None
            if fmt == "mp3":
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    mp3_path = f.name
                # 320 kbps when mastered so the polish survives the codec;
                # 192 kbps when unmastered (the v58 default).
                mp3_bitrate = "320k" if master else "192k"
                wav_to_mp3(wav_for_primary, mp3_path, bitrate=mp3_bitrate)
                primary_local = mp3_path
                primary_ct = "audio/mpeg"
            else:
                primary_local = wav_for_primary
                primary_ct = "audio/wav"

            audio_url = upload_to_supabase(primary_local, storage_path, primary_ct)

            resp = {
                "audio_url": audio_url,
                "format": fmt,
                "storage_path": storage_path,
                "duration": duration,
                "task": "text2music",
            }
            if master:
                resp["mastered"] = True
                resp["mastered_lufs_in"] = mastered_stats.get("input_i")
                resp["mastered_lufs_out"] = -14.0
                resp["mastered_tp_in"] = mastered_stats.get("input_tp")
                resp["mp3_bitrate"] = "320k" if fmt == "mp3" else None

            if keep_wav:
                # v60 — keep_wav always saves the RAW unmastered render,
                # never the mastered intermediate. The artist's "lossless
                # source for my own mastering pipeline" expectation only
                # holds if we preserve the unprocessed audio. The mastered
                # MP3 (audio_url) covers the "I just want it to sound
                # polished now" case.
                wav_bytes = os.path.getsize(rendered_wav)
                # v57 — Long songs blow past Supabase's per-object cap.
                # The primary MP3 has already uploaded by the time we
                # get here, so a 413 on the WAV shouldn't lose us the
                # whole job. Pre-check the size; fall back to a graceful
                # response so the caller (which gets the song from
                # audio_url) is unaffected.
                if wav_bytes > SUPABASE_MAX_OBJECT_BYTES:
                    print(
                        f"[v57 wav-skip] wav_bytes={wav_bytes} "
                        f"cap={SUPABASE_MAX_OBJECT_BYTES} — "
                        "skipping WAV upload (over per-object cap). "
                        "MP3 already uploaded; song is still usable."
                    )
                    resp["wav_skipped"] = True
                    resp["wav_skipped_reason"] = "size_exceeds_supabase_cap"
                    resp["wav_bytes"] = wav_bytes
                    resp["wav_storage_path"] = storage_path_wav  # for debugging
                else:
                    try:
                        wav_url = upload_to_supabase(
                            rendered_wav, storage_path_wav, "audio/wav"
                        )
                        resp["wav_url"] = wav_url
                        resp["wav_storage_path"] = storage_path_wav
                    except Exception as wav_err:
                        # Belt-and-suspenders: if the cap check was wrong
                        # (operator bumped Supabase but didn't bump our
                        # env), still don't lose the job.
                        print(
                            f"[v57 wav-skip] upload raised "
                            f"({type(wav_err).__name__}: {wav_err}). "
                            "MP3 already saved; reporting skip."
                        )
                        resp["wav_skipped"] = True
                        resp["wav_skipped_reason"] = (
                            f"upload_error:{type(wav_err).__name__}"
                        )
                        resp["wav_bytes"] = wav_bytes
                        resp["wav_storage_path"] = storage_path_wav

            if return_b64:
                with open(primary_local, "rb") as f:
                    resp["audio_b64"] = base64.b64encode(f.read()).decode("utf-8")

            if mp3_path and os.path.exists(mp3_path):
                try: os.unlink(mp3_path)
                except: pass

            return resp

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        if src_temp and os.path.exists(src_temp):
            try: os.unlink(src_temp)
            except: pass
        if lora_temp and os.path.exists(lora_temp):
            try: os.unlink(lora_temp)
            except: pass

runpod.serverless.start({"handler": handler})
