#!/usr/bin/env python3
"""
yt-dlp -> Clips AI pipeline: turn a long video URL into vertical short clips.

Stages
------
1. download   : fetch the video with yt-dlp (uses this repo's source when run from it)
2. transcribe : WhisperX transcription via clipsai.Transcriber
3. find clips : topic segmentation via clipsai.ClipFinder
4. reframe    : optional speaker-tracking crop via clipsai.resize (needs a HF token)
5. render     : cut each clip with ffmpeg, following the speaker if reframing is on
6. publish    : optional upload to Postiz as drafts (--postiz)

Every stage is optional-ish: without a HuggingFace token you still get accurate
horizontal cuts, and --dry-run stops after stage 3 so you can inspect the clip
list before spending time on encoding.

Usage
-----
    python shorts/make_shorts.py "https://www.youtube.com/watch?v=..." \
        --output-dir shorts_out \
        --hf-token hf_xxx

See shorts/README.md for install notes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postiz_client import PostizClient, PostizError  # noqa: E402

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def log(msg: str) -> None:
    print(f"[shorts] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[shorts] error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            die(
                f"{tool} not found on PATH. Install it first "
                "(apt install ffmpeg / brew install ffmpeg)."
            )


def slugify(text: str, limit: int = 60) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in text]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:limit] or "video"


def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


# --------------------------------------------------------------------------- #
# stage 1 - download
# --------------------------------------------------------------------------- #


@dataclass
class Source:
    path: Path
    title: str
    duration: float | None = None
    width: int | None = None
    height: int | None = None


def import_yt_dlp():
    """Import yt_dlp, preferring this repo's own source over any installed copy."""
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "yt_dlp" / "__init__.py").exists():
        sys.path.insert(0, str(repo_root))
    try:
        import yt_dlp
    except ImportError as exc:
        die(
            "could not import yt_dlp. Either run this script from inside a "
            "yt-dlp checkout, or `pip install -U yt-dlp`.\n"
            f"(underlying import error: {exc})"
        )
    return yt_dlp


def download(url: str, work_dir: Path, fmt: str, cookies_from_browser: str | None) -> Source:
    """Download `url` with yt-dlp and return the resulting file."""
    yt_dlp = import_yt_dlp()

    work_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(work_dir / "%(title).100B [%(id)s].%(ext)s")

    opts: dict = {
        "format": fmt,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noprogress": False,
        "quiet": False,
        "no_warnings": False,
        # clipsai works on a local file, so make sure we end up with one real file
        "postprocessors": [],
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    log(f"downloading {url}")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info:  # playlist -> take the first entry
                entries = [e for e in info["entries"] if e]
                if not entries:
                    die("playlist contained no downloadable entries")
                info = entries[0]
            path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as exc:
        die(f"yt-dlp could not download that URL:\n  {exc}")
    except KeyboardInterrupt:
        die("interrupted during download", code=130)

    if not path.exists():
        # merged output may have a different container than prepare_filename guessed
        candidates = sorted(
            work_dir.glob(f"*[{info.get('id', '')}]*"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not candidates:
            die(f"download finished but no file found in {work_dir}")
        path = candidates[0]

    log(f"downloaded -> {path.name}")
    return Source(
        path=path.resolve(),
        title=info.get("title") or path.stem,
        duration=info.get("duration"),
        width=info.get("width"),
        height=info.get("height"),
    )


def probe_dimensions(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        die(f"ffprobe failed on {path.name}: {out.stderr.strip()}")
    w, h = out.stdout.strip().split("x")[:2]
    return int(w), int(h)


# --------------------------------------------------------------------------- #
# stage 2 + 3 - transcribe and find clips
# --------------------------------------------------------------------------- #


@dataclass
class ClipSpec:
    index: int
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


CLIPSAI_INSTALL_HINT = (
    "clipsai is not installed. Install it with:\n"
    "    pip install clipsai\n"
    "    pip install whisperx@git+https://github.com/m-bain/whisperx.git"
)


def _register_surrogate_clipsai() -> None:
    """Put a stand-in `clipsai` package in sys.modules so its __init__ never runs.

    Importing clipsai.transcribe.transcriber normally executes the parent
    package's __init__ first, and that __init__ imports the resize code (and so
    facenet-pytorch). Registering a surrogate parent with the real __path__ lets
    the submodules import exactly as usual while skipping those re-exports.
    Safe here because no clipsai module imports from the package root.
    """
    import importlib.util
    import types

    if "clipsai" in sys.modules:
        return
    try:
        spec = importlib.util.find_spec("clipsai")  # does not execute __init__
    except Exception:
        return
    if spec is None or not spec.submodule_search_locations:
        return  # not installed - let the real import raise a clear error

    surrogate = types.ModuleType("clipsai")
    surrogate.__path__ = list(spec.submodule_search_locations)
    surrogate.__spec__ = spec
    sys.modules["clipsai"] = surrogate


def import_clip_tools():
    """Return (Transcriber, ClipFinder), imported from their own submodules.

    Deliberately NOT `import clipsai`. The package's __init__ eagerly imports
    the resize code, which needs facenet-pytorch; that is pinned to numpy<2,
    while whisperx and pyannote.audio need numpy>=2.1. Those cannot coexist, so
    importing the package wholesale fails on any working install. The transcribe
    and clip modules have no facenet dependency, so they import cleanly.
    """
    _register_surrogate_clipsai()
    try:
        from clipsai.transcribe.transcriber import Transcriber
        from clipsai.clip.clipfinder import ClipFinder
    except ImportError as exc:
        die(f"{CLIPSAI_INSTALL_HINT}\n(underlying import error: {exc})")
    return Transcriber, ClipFinder


def import_resize():
    """Return (resize_callable, None), or (None, exception) if unavailable.

    Reframing is optional: on a numpy>=2 install facenet-pytorch usually cannot
    load, and that must not take the rest of the pipeline down with it.
    """
    _register_surrogate_clipsai()
    try:
        from clipsai.resize.resize import resize
    except Exception as exc:  # ImportError, or numpy/torch ABI errors
        return None, exc
    return resize, None


def find_clips(source: Source, args) -> tuple[list[ClipSpec], object]:
    Transcriber, ClipFinder = import_clip_tools()

    log(f"transcribing with whisper model '{args.whisper_model}' (this is the slow part)")
    transcriber = Transcriber(model_size=args.whisper_model, device=args.device)
    transcription = transcriber.transcribe(audio_file_path=str(source.path))

    log("finding clip boundaries")
    clipfinder = ClipFinder(
        device=args.device,
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
        cutoff_policy=args.cutoff_policy,
    )
    found = clipfinder.find_clips(transcription=transcription)
    log(f"clipsai proposed {len(found)} clip(s)")

    full_text = ""
    try:
        full_text = transcription.text or ""
    except Exception:  # pragma: no cover - defensive across clipsai versions
        full_text = ""

    specs: list[ClipSpec] = []
    for i, clip in enumerate(found, start=1):
        text = ""
        if full_text:
            try:
                text = full_text[clip.start_char : clip.end_char].strip()
            except Exception:
                text = ""
        specs.append(
            ClipSpec(
                index=i,
                start=float(clip.start_time),
                end=float(clip.end_time),
                text=text,
            )
        )

    if args.max_clips:
        specs = specs[: args.max_clips]
    return specs, transcription


# --------------------------------------------------------------------------- #
# stage 4 - reframe
# --------------------------------------------------------------------------- #


@dataclass
class Reframe:
    crop_width: int
    crop_height: int
    original_width: int
    original_height: int
    segments: list[dict] = field(default_factory=list)


def compute_reframe(source: Source, args) -> Reframe | None:
    """Run clipsai.resize to get speaker-following crop windows, or None."""
    token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get(
        "PYANNOTE_AUTH_TOKEN"
    )
    if args.no_reframe:
        log("reframing disabled (--no-reframe); keeping the original aspect ratio")
        return None
    if not token:
        log(
            "no HuggingFace token found, so skipping speaker-tracking reframe. "
            "Pass --hf-token or set HUGGINGFACE_TOKEN to enable it, or use "
            "--center-crop for a static vertical crop."
        )
        return None

    resize, resize_error = import_resize()
    if resize is None:
        log(
            "speaker-tracking reframe unavailable: clipsai's resize module could "
            f"not be loaded ({type(resize_error).__name__}: {resize_error})."
        )
        log(
            "  This is usually facenet-pytorch, which pins numpy<2 while whisperx "
            "needs numpy>=2.1. Use --center-crop for a static vertical crop."
        )
        return None

    w, h = args.aspect_ratio
    log(f"running speaker diarization + face detection for a {w}:{h} reframe")
    crops = resize(
        video_file_path=str(source.path),
        pyannote_auth_token=token,
        aspect_ratio=(w, h),
        device=args.device,
    )
    segments = [
        {
            "start_time": float(s.start_time),
            "end_time": float(s.end_time),
            "x": int(s.x),
            "y": int(s.y),
            "speakers": list(s.speakers),
        }
        for s in crops.segments
    ]
    log(f"got {len(segments)} crop segment(s), crop box {crops.crop_width}x{crops.crop_height}")
    return Reframe(
        crop_width=int(crops.crop_width),
        crop_height=int(crops.crop_height),
        original_width=int(crops.original_width),
        original_height=int(crops.original_height),
        segments=segments,
    )


def center_crop_box(src_w: int, src_h: int, ratio: tuple[int, int]) -> tuple[int, int, int, int]:
    rw, rh = ratio
    crop_h = src_h
    crop_w = int(round(src_h * rw / rh))
    if crop_w > src_w:  # source is already narrower than the target ratio
        crop_w = src_w
        crop_h = int(round(src_w * rh / rw))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    return crop_w, crop_h, (src_w - crop_w) // 2, (src_h - crop_h) // 2


def crop_x_expression(reframe: Reframe, clip: ClipSpec) -> str:
    """
    Build an ffmpeg expression that moves the crop window as the speaker changes.

    Clip rendering seeks with -ss before -i, so the output timestamp `t` starts at
    0 while the segment times are absolute in the source: compare against
    (segment_end - clip.start).
    """
    max_x = max(0, reframe.original_width - reframe.crop_width)

    relevant = [
        s for s in reframe.segments
        if s["end_time"] > clip.start and s["start_time"] < clip.end
    ]
    if not relevant:
        return str(max_x // 2)

    def clamp(x: int) -> int:
        return max(0, min(int(x), max_x))

    # Innermost fallback is the last overlapping segment.
    expr = str(clamp(relevant[-1]["x"]))
    for seg in reversed(relevant[:-1]):
        boundary = max(0.0, seg["end_time"] - clip.start)
        expr = f"if(lt(t,{boundary:.3f}),{clamp(seg['x'])},{expr})"
    return expr


# --------------------------------------------------------------------------- #
# stage 5 - render
# --------------------------------------------------------------------------- #


def render_clip(
    source: Source,
    clip: ClipSpec,
    out_path: Path,
    reframe: Reframe | None,
    args,
    src_dims: tuple[int, int],
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-ss", hms(clip.start),
        "-t", hms(clip.duration),
        "-i", str(source.path),
    ]

    vf: list[str] = []
    if reframe is not None:
        x_expr = crop_x_expression(reframe, clip)
        y = max(0, (reframe.original_height - reframe.crop_height) // 2)
        vf.append(
            f"crop={reframe.crop_width}:{reframe.crop_height}:'{x_expr}':{y}"
        )
    elif args.center_crop:
        cw, ch, cx, cy = center_crop_box(*src_dims, args.aspect_ratio)
        vf.append(f"crop={cw}:{ch}:{cx}:{cy}")

    if args.scale_height and vf:
        vf.append(f"scale=-2:{args.scale_height}")

    if vf:
        cmd += ["-vf", ",".join(vf), "-c:v", "libx264", "-crf", str(args.crf),
                "-preset", args.preset, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"]
    elif args.copy:
        cmd += ["-c", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"]

    cmd += ["-movflags", "+faststart", str(out_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        die(f"ffmpeg failed on clip {clip.index}:\n{tail}")


# --------------------------------------------------------------------------- #
# stage 6 - publish to Postiz
# --------------------------------------------------------------------------- #


def resolve_channels(client: PostizClient, wanted: list[str]) -> list[dict]:
    """Match --postiz-channels values against Postiz integrations."""
    available = client.integrations()
    if not available:
        die("Postiz returned no channels: connect at least one in Postiz first")

    enabled = [c for c in available if not c.get("disabled")]
    if not wanted:
        if not enabled:
            die("every Postiz channel is disabled; nothing to post to")
        return enabled

    by_key: dict[str, dict] = {}
    for channel in available:
        for key in (channel.get("id"), channel.get("name"), channel.get("identifier")):
            if key:
                by_key.setdefault(str(key).strip().lower(), channel)

    chosen: list[dict] = []
    for name in wanted:
        channel = by_key.get(name.strip().lower())
        if channel is None:
            known = ", ".join(
                f"{c.get('name')} [{c.get('identifier')}]" for c in available
            )
            die(f"no Postiz channel matching '{name}'. Available: {known}")
        if channel.get("disabled"):
            die(f"Postiz channel '{channel.get('name')}' is disabled")
        if channel not in chosen:
            chosen.append(channel)
    return chosen


def build_caption(template: str, clip: ClipSpec, source: Source, total: int, limit: int) -> str:
    excerpt = " ".join(clip.text.split())
    if limit and len(excerpt) > limit:
        excerpt = excerpt[:limit].rsplit(" ", 1)[0] + "..."
    caption = template.format(
        title=source.title,
        index=clip.index,
        total=total,
        transcript=excerpt,
        start=hms(clip.start),
        duration=f"{clip.duration:.0f}",
        url="",
    )
    return caption.strip() or source.title


def publish_to_postiz(
    clips: list[ClipSpec],
    rendered: dict[int, Path],
    source: Source,
    args,
) -> list[dict]:
    key = args.postiz_key or os.environ.get("POSTIZ_API_KEY")
    if not key:
        die("--postiz needs an API key: pass --postiz-key or set POSTIZ_API_KEY")
    base_url = args.postiz_url or os.environ.get("POSTIZ_API_URL") or "https://api.postiz.com"

    client = PostizClient(api_key=key, base_url=base_url)
    channels = resolve_channels(client, args.postiz_channels)
    log(
        f"posting to {len(channels)} Postiz channel(s) as "
        f"'{args.postiz_type}': "
        + ", ".join(f"{c.get('name')} [{c.get('identifier')}]" for c in channels)
    )

    # Work out the first slot, then space the rest out evenly.
    if args.postiz_start:
        try:
            first = datetime.fromisoformat(args.postiz_start.replace("Z", "+00:00"))
        except ValueError:
            die(f"could not parse --postiz-start '{args.postiz_start}' (use ISO 8601)")
        if first.tzinfo is None:
            first = first.replace(tzinfo=timezone.utc)
    else:
        first = None
        slot = client.find_slot(str(channels[0]["id"]))
        if slot:
            try:
                first = datetime.fromisoformat(str(slot).replace("Z", "+00:00"))
                log(f"Postiz suggested the next free slot: {first.isoformat()}")
            except ValueError:
                first = None
        if first is None:
            first = datetime.now(timezone.utc) + timedelta(hours=1)
    if first.tzinfo is None:
        first = first.replace(tzinfo=timezone.utc)

    results: list[dict] = []
    for position, clip in enumerate(clips):
        clip_path = rendered.get(clip.index)
        if clip_path is None or not clip_path.exists():
            log(f"  clip {clip.index}: no rendered file, skipping Postiz upload")
            continue

        when = first + timedelta(hours=args.postiz_interval_hours * position)
        caption = build_caption(
            args.postiz_caption, clip, source, len(clips), args.postiz_caption_limit
        )

        try:
            log(f"  uploading clip {clip.index} ({clip_path.stat().st_size // 1024} KiB)")
            media = client.upload(clip_path)
            log(f"  creating {args.postiz_type} for clip {clip.index} at {when.isoformat()}")
            created = client.create_post(
                integration_ids=[str(c["id"]) for c in channels],
                content=caption,
                media=[media],
                date_iso=when.isoformat(),
                post_type=args.postiz_type,
            )
        except PostizError as exc:
            die(
                f"Postiz rejected clip {clip.index}: {exc}\n"
                "Clips already uploaded stay in your Postiz media library."
            )

        post_ids: list[str] = []
        if isinstance(created, list):
            post_ids = [str(p.get("id")) for p in created if isinstance(p, dict) and p.get("id")]
        elif isinstance(created, dict):
            if created.get("id"):
                post_ids = [str(created["id"])]

        results.append(
            {
                "clip": clip.index,
                "media_id": media.get("id"),
                "media_path": media.get("path"),
                "scheduled_for": when.isoformat(),
                "type": args.postiz_type,
                "post_ids": post_ids,
                "channels": [c.get("name") for c in channels],
            }
        )

    return results


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def parse_ratio(text: str) -> tuple[int, int]:
    sep = ":" if ":" in text else "x"
    try:
        w, h = (int(p) for p in text.split(sep, 1))
    except ValueError:
        raise argparse.ArgumentTypeError("aspect ratio must look like 9:16")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("aspect ratio parts must be positive")
    return w, h


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_shorts.py",
        description="Download a video with yt-dlp and cut it into shorts with Clips AI.",
    )
    p.add_argument("url", help="video URL to pass to yt-dlp")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("shorts_out"),
                   help="where clips and the manifest are written (default: shorts_out)")
    p.add_argument("--format", default="bestvideo*+bestaudio/best",
                   help="yt-dlp format selector")
    p.add_argument("--cookies-from-browser", default=None,
                   help="yt-dlp browser name for cookies, e.g. chrome")
    p.add_argument("--delete-source", action="store_true",
                   help="delete the downloaded source file once clips are rendered")

    p.add_argument("--whisper-model", default="large-v2",
                   help="whisper/whisperx model size (default: large-v2; try 'base' to go fast)")
    p.add_argument("--device", default=None, help="'cpu' or 'cuda' (default: auto)")
    p.add_argument("--min-duration", type=int, default=15,
                   help="minimum clip length in seconds (default: 15)")
    p.add_argument("--max-duration", type=int, default=90,
                   help="maximum clip length in seconds (default: 90)")
    p.add_argument("--cutoff-policy", default="high", choices=["average", "high", "low"],
                   help="how aggressively clipsai splits topics (default: high)")
    p.add_argument("--max-clips", type=int, default=0,
                   help="render at most N clips (0 = all)")

    p.add_argument("--aspect-ratio", type=parse_ratio, default=(9, 16),
                   help="target aspect ratio (default: 9:16)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token for pyannote (or set HUGGINGFACE_TOKEN)")
    p.add_argument("--no-reframe", action="store_true",
                   help="skip speaker-tracking reframing entirely")
    p.add_argument("--center-crop", action="store_true",
                   help="static centered vertical crop when reframing is unavailable")

    p.add_argument("--crf", type=int, default=20, help="x264 quality, lower is better (default: 20)")
    p.add_argument("--preset", default="veryfast", help="x264 preset (default: veryfast)")
    p.add_argument("--scale-height", type=int, default=0,
                   help="scale cropped output to this height, e.g. 1920 (0 = no scaling)")
    p.add_argument("--copy", action="store_true",
                   help="stream-copy instead of re-encoding when not cropping (fast, cuts land on keyframes)")
    p.add_argument("--dry-run", action="store_true",
                   help="download and list clips, write the manifest, render nothing")

    postiz = p.add_argument_group("Postiz publishing")
    postiz.add_argument("--postiz", action="store_true",
                        help="upload rendered clips to Postiz and create posts")
    postiz.add_argument("--postiz-url", default=None,
                        help="Postiz API base (default: https://api.postiz.com; "
                             "self-hosted is usually https://your-domain/api). "
                             "Env: POSTIZ_API_URL")
    postiz.add_argument("--postiz-key", default=None,
                        help="Postiz API key (env: POSTIZ_API_KEY)")
    postiz.add_argument("--postiz-channels", default="",
                        type=lambda s: [v for v in (p.strip() for p in s.split(",")) if v],
                        help="comma-separated channel ids or names (default: all enabled)")
    postiz.add_argument("--postiz-type", default="draft",
                        choices=["draft", "schedule", "now"],
                        help="draft (default, nothing goes live), schedule, or now")
    postiz.add_argument("--postiz-start", default=None,
                        help="ISO 8601 time for the first post "
                             "(default: ask Postiz for the next free slot)")
    postiz.add_argument("--postiz-interval-hours", type=float, default=24.0,
                        help="hours between consecutive clips (default: 24)")
    postiz.add_argument("--postiz-caption",
                        default="{title}\n\n{transcript}",
                        help="caption template; fields: {title} {index} {total} "
                             "{transcript} {start} {duration}")
    postiz.add_argument("--postiz-caption-limit", type=int, default=400,
                        help="truncate {transcript} to this many characters (default: 400)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_ffmpeg()

    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = download(args.url, out_dir / "source", args.format, args.cookies_from_browser)
    src_dims = probe_dimensions(source.path)

    clips, _transcription = find_clips(source, args)
    if not clips:
        log("clipsai found no clips: try a lower --min-duration or --cutoff-policy low")
        return 2

    for c in clips:
        log(f"  clip {c.index}: {hms(c.start)} -> {hms(c.end)}  ({c.duration:.1f}s)")

    reframe = None if args.dry_run else compute_reframe(source, args)

    slug = slugify(source.title)
    manifest = {
        "url": args.url,
        "title": source.title,
        "source_file": str(source.path),
        "source_dimensions": {"width": src_dims[0], "height": src_dims[1]},
        "aspect_ratio": list(args.aspect_ratio),
        "reframed": reframe is not None,
        "clips": [],
    }

    rendered: dict[int, Path] = {}
    for clip in clips:
        name = f"{slug}-clip{clip.index:02d}.mp4"
        out_path = out_dir / name
        entry = {
            "index": clip.index,
            "start": round(clip.start, 3),
            "end": round(clip.end, 3),
            "duration": round(clip.duration, 3),
            "file": None if args.dry_run else str(out_path),
            "transcript": clip.text,
        }
        if not args.dry_run:
            log(f"rendering clip {clip.index}/{len(clips)} -> {name}")
            render_clip(source, clip, out_path, reframe, args, src_dims)
            rendered[clip.index] = out_path
        manifest["clips"].append(entry)

    if args.postiz:
        if args.dry_run:
            log("dry run: skipping Postiz upload")
        else:
            manifest["postiz"] = publish_to_postiz(clips, rendered, source, args)

    manifest_path = out_dir / "clips.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"manifest -> {manifest_path}")

    if args.delete_source and not args.dry_run:
        source.path.unlink(missing_ok=True)
        log("deleted source file")

    if args.dry_run:
        log("dry run: no clips rendered")
    else:
        log(f"done: {len(clips)} clip(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
