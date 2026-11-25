# yt-dlp → Clips AI → Postiz shorts pipeline

`make_shorts.py` takes a video URL and gives back vertical short clips, and
optionally puts them straight into your Postiz posting calendar:

```
URL ──yt-dlp──> video file ──WhisperX──> transcript ──ClipFinder──> clip ranges
                                                           │
                              pyannote + face detect ──────┴──> crop windows
                                                                     │
                                                      ffmpeg ────────┴──> clips + clips.json
                                                                     │
                                                       Postiz ───────┴──> drafts / scheduled posts
```

## Install

Python 3.9–3.12 is the safe range. Clips AI pulls in PyTorch, so expect a
multi-gigabyte install.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r shorts/requirements.txt
pip install whisperx@git+https://github.com/m-bain/whisperx.git
```

System packages:

- **ffmpeg** and **ffprobe** — `apt install ffmpeg` / `brew install ffmpeg`
- **libmagic** — `apt install libmagic1` / `brew install libmagic`
  (Windows: `pip install python-magic-bin`)

yt-dlp itself comes from this repo when you run the script from the repo root.

## Reframing needs a HuggingFace token

The speaker-tracking 9:16 crop uses pyannote for diarization, which needs a
free token and model access:

1. Create a token at <https://huggingface.co/settings/tokens>
2. Accept the terms on `pyannote/speaker-diarization` and `pyannote/segmentation`
3. `export HUGGINGFACE_TOKEN=hf_...` or pass `--hf-token`

Without a token the script still runs and produces accurate cuts in the
original aspect ratio. Add `--center-crop` for a static vertical crop instead.

## Use

Plan first, render nothing:

```bash
python shorts/make_shorts.py "https://youtu.be/VIDEO_ID" --dry-run --whisper-model base
```

Full run with reframing:

```bash
python shorts/make_shorts.py "https://youtu.be/VIDEO_ID" \
    --output-dir shorts_out \
    --hf-token "$HUGGINGFACE_TOKEN" \
    --min-duration 20 --max-duration 75 \
    --scale-height 1920
```

Fast, no GPU, no token:

```bash
python shorts/make_shorts.py "https://youtu.be/VIDEO_ID" \
    --whisper-model base --no-reframe --center-crop --max-clips 3
```

## Options worth knowing

| Flag | What it does |
| --- | --- |
| `--whisper-model` | `large-v2` is the default and the slow part. `base` is roughly 10× faster and fine for finding boundaries. |
| `--device` | `cuda` if you have an NVIDIA GPU, otherwise `cpu`. Auto-detected by default. |
| `--min-duration` / `--max-duration` | Clip length bounds in seconds. |
| `--cutoff-policy` | `low` splits more often, `high` splits less. |
| `--max-clips` | Render only the first N clips. |
| `--center-crop` | Static centered vertical crop, no ML, no token. |
| `--scale-height 1920` | Normalize output to 1080×1920. |
| `--copy` | Stream-copy instead of re-encoding when not cropping. Fast, but cuts snap to keyframes. |
| `--dry-run` | Stop after listing clips and writing `clips.json`. |

## Publishing to Postiz

[Postiz](https://github.com/gitroomhq/postiz-app) is an open-source social
media scheduler. With `--postiz` the pipeline uploads each rendered clip and
creates a post for it. Only the stdlib is used for this — no extra packages.

```bash
export POSTIZ_API_KEY=your-key
export POSTIZ_API_URL=https://social.example.com/api   # self-hosted
# Postiz Cloud needs no URL; it defaults to https://api.postiz.com

python shorts/make_shorts.py "https://youtu.be/VIDEO_ID" \
    --postiz --postiz-channels "My TikTok,My YouTube"
```

Get the API key from Postiz under **Settings → Public API**. The key goes in
the `Authorization` header as-is, with no `Bearer` prefix.

**Posts are created as drafts by default and nothing goes live.** That is
deliberate: `--postiz-type schedule` or `now` makes Postiz run its
per-provider validation, and channels like YouTube and TikTok reject a post
that carries no title or privacy settings of their own. Review the drafts in
Postiz, fill in what each channel wants, then schedule from there.

| Flag | What it does |
| --- | --- |
| `--postiz` | Turn on publishing. |
| `--postiz-url` | API base. Cloud: `https://api.postiz.com`. Self-hosted: `https://your-domain/api`. Env `POSTIZ_API_URL`. |
| `--postiz-key` | API key. Env `POSTIZ_API_KEY`. |
| `--postiz-channels` | Comma-separated channel ids, names or provider identifiers. Matching is case-insensitive. Default: every enabled channel. |
| `--postiz-type` | `draft` (default), `schedule`, or `now`. |
| `--postiz-start` | ISO 8601 time for the first post. Default: ask Postiz for its next free slot. |
| `--postiz-interval-hours` | Spacing between clips, default 24. |
| `--postiz-caption` | Template. Fields: `{title}` `{index}` `{total}` `{transcript}` `{start}` `{duration}`. |
| `--postiz-caption-limit` | Truncate `{transcript}` at this length, default 400. |

`--dry-run` skips Postiz entirely, so nothing is uploaded while you are still
tuning clip boundaries.

## Output

```
shorts_out/
├── source/My Video [abc123].mp4
├── My-Video-clip01.mp4
├── My-Video-clip02.mp4
└── clips.json
```

`clips.json` records each clip's start, end, duration, output path and the
transcript text for that range — handy for captions or titles. After a
`--postiz` run it also carries a `postiz` block with the media id, post ids,
target channels and scheduled time for every clip.

## Notes and limits

- Clips AI is built for speech-driven video (podcasts, interviews, talks). On
  music or heavy b-roll the topic segmentation has little to work with.
- The reframe crop follows the active speaker by switching the crop window at
  segment boundaries. The switch is a cut, not a pan.
- Clips AI's last upstream release is 0.2.1 (Jan 2024). It pins older PyTorch
  and WhisperX behaviour, which is why a dedicated virtualenv is worth it.
- Postiz is AGPL-3.0. Calling its HTTP API from this script does not make this
  code a derivative work, but if you redistribute a modified Postiz itself the
  licence applies to that.
- Respect the terms of the site you download from and the rights of the video
  you are clipping.
