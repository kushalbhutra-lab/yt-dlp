<#
.SYNOPSIS
    Sets up the yt-dlp -> ClipsAI -> Postiz clipping pipeline on Windows.

.DESCRIPTION
    Creates clipping-pipeline\ containing:
      yt-dlp\      your fork on the shorts-pipeline branch (brings make_shorts.py)
      clipsai\     upstream ClipsAI source, for reference
      postiz-app\  upstream Postiz source, to run with Docker
      .venv\       Python 3.9-3.12 environment with clipsai + whisperx

.PARAMETER Check
    Only check prerequisites and report. Changes nothing.

.PARAMETER Root
    Where to create the project. Defaults to .\clipping-pipeline

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_pipeline.ps1 -Check

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_pipeline.ps1
#>

[CmdletBinding()]
param(
    [switch]$Check,
    [string]$Root = (Join-Path (Get-Location) 'clipping-pipeline')
)

# Native tools write to stderr routinely; we gate on $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'

$Fork   = 'https://github.com/kushalbhutra-lab/yt-dlp.git'
$Branch = 'shorts-pipeline'

function Write-Ok   ($m) { Write-Host "  ok    $m"   -ForegroundColor Green }
function Write-Warn ($m) { Write-Host "  warn  $m"   -ForegroundColor Yellow }
function Write-Bad  ($m) { Write-Host "  fail  $m"   -ForegroundColor Red }
function Write-Step ($m) { Write-Host "`n==> $m"     -ForegroundColor Cyan }
function Fail       ($m) { Write-Host "`nerror: $m"  -ForegroundColor Red; exit 1 }

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# Windows ships stub python.exe / python3.exe under WindowsApps that only open
# the Microsoft Store. They answer Get-Command but are not a usable Python.
function Test-RealCommand($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) { return $false }
    $src = $cmd.Source
    if ($src -and $src -like '*\WindowsApps\*') { return $false }
    return $true
}

# ------------------------------------------------------------- prerequisites #
Write-Step 'Checking prerequisites'
$missing = $false

if (-not (Test-Command winget)) {
    Write-Warn 'winget not found - install the three tools manually:'
    Write-Warn '  python  https://www.python.org/downloads/release/python-3129/'
    Write-Warn '  git     https://git-scm.com/download/win'
    Write-Warn '  ffmpeg  https://www.gyan.dev/ffmpeg/builds/  (add bin\ to PATH)'
}

if (Test-Command git) {
    Write-Ok ("git " + ((git --version) -split ' ')[2])
} else {
    Write-Bad 'git not found'
    Write-Warn '  install with:  winget install --id Git.Git -e'
    $missing = $true
}

if ((Test-Command ffmpeg) -and (Test-Command ffprobe)) {
    $ffver = (& ffmpeg -version 2>$null | Select-Object -First 1)
    Write-Ok ("ffmpeg found - " + $ffver)
} else {
    Write-Bad 'ffmpeg/ffprobe not found - clips cannot be cut without them'
    Write-Warn '  install with:  winget install --id Gyan.FFmpeg -e'
    Write-Warn '  then close and reopen PowerShell so PATH refreshes'
    $missing = $true
}

# ClipsAI (last release Jan 2024) pins older torch/whisperx; 3.13+ breaks it.
# Resolve a concrete interpreter path, so later calls are a plain `& $exe ...`
$pythonExe = $null
if (Test-RealCommand py) {
    foreach ($v in '3.12', '3.11', '3.10', '3.9') {
        $cand = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $cand) {
            $first = ($cand | Where-Object { $_ -is [string] } | Select-Object -First 1)
            if ($first) { $pythonExe = $first.Trim(); break }
        }
    }
}
if (-not $pythonExe -and (Test-RealCommand python)) {
    $info = @(& python -c "import sys; print('%d.%d' % sys.version_info[:2]); print(sys.executable)" 2>$null)
    $info = @($info | Where-Object { $_ -is [string] -or $_ -isnot [System.Management.Automation.ErrorRecord] })
    if ($info.Count -ge 2) {
        $pv = $info[0].Trim()
        if ([version]$pv -ge [version]'3.9' -and [version]$pv -lt [version]'3.13') {
            $pythonExe = $info[1].Trim()
        } else {
            Write-Bad "python on PATH is $pv - ClipsAI needs 3.9 to 3.12"
        }
    }
}
if (-not $pythonExe) {
    if ((Test-Command python) -and -not (Test-RealCommand python)) {
        Write-Bad 'only the Microsoft Store placeholder for python.exe is on PATH'
    } else {
        Write-Bad 'no Python 3.9-3.12 found'
    }
    Write-Warn '  install with:  winget install --id Python.Python.3.12 -e'
    Write-Warn '  then close and reopen PowerShell'
    $missing = $true
} else {
    $pvFull = & $pythonExe -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
    Write-Ok ("python " + $pvFull + "  -> " + $pythonExe)
}

if (Test-Command docker) {
    & docker info *> $null
    if ($LASTEXITCODE -eq 0) { Write-Ok 'docker running (for Postiz)' }
    else { Write-Warn 'docker installed but not running - start Docker Desktop before launching Postiz' }
} else {
    Write-Warn 'docker not found - needed only to run Postiz itself, not to cut clips'
    Write-Warn '  install with:  winget install --id Docker.DockerDesktop -e'
}

# Node is not needed for clipping, nor for Postiz via Docker. Only for Postiz dev.
if (Test-Command node) {
    Write-Ok ("node " + (node --version))
} else {
    Write-Warn 'node not found - fine for clipping and for running Postiz in Docker'
    Write-Warn '  only needed to develop Postiz itself:  winget install --id OpenJS.NodeJS.LTS -e'
}

$gpu = $false
if (Test-Command nvidia-smi) {
    & nvidia-smi *> $null
    if ($LASTEXITCODE -eq 0) { Write-Ok 'NVIDIA GPU detected - transcription can use CUDA'; $gpu = $true }
}
if (-not $gpu) { Write-Warn 'no NVIDIA GPU detected - use --whisper-model base or expect slow transcription' }

if ($missing) { Fail 'fix the failures above, then re-run this script.' }
Write-Host "`nPrerequisites look good." -ForegroundColor Green
if ($Check) { Write-Host '-Check given, stopping here.' -ForegroundColor DarkGray; exit 0 }

# ---------------------------------------------------------------- clone repos #
Write-Step "Setting up $Root"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
Set-Location $Root

function Get-Repo($name, $url, $branch) {
    $target = Join-Path $Root $name
    if (Test-Path (Join-Path $target '.git')) {
        Write-Ok "$name already cloned, leaving it alone"
        return
    }
    if (Test-Path $target) {
        Fail "$target exists but is not a git clone - move it aside and re-run"
    }
    Write-Host "  cloning $name ..."
    if ($branch) { & git clone --depth 1 --branch $branch $url $target *> $null }
    else         { & git clone --depth 1 $url $target *> $null }
    if ($LASTEXITCODE -ne 0) { Fail "failed to clone $url" }
    Write-Ok "$name cloned"
}

Get-Repo 'yt-dlp'     $Fork $Branch
Get-Repo 'clipsai'    'https://github.com/ClipsAI/clipsai.git' $null
Get-Repo 'postiz-app' 'https://github.com/gitroomhq/postiz-app.git' $null

$script = Join-Path $Root 'yt-dlp\shorts\make_shorts.py'
if (-not (Test-Path $script)) {
    Fail "yt-dlp\shorts\make_shorts.py missing - is branch $Branch pushed to your fork?"
}
Write-Ok 'pipeline script present at yt-dlp\shorts\make_shorts.py'

# ---------------------------------------------------------------- python deps #
Write-Step 'Building the Python environment (this pulls PyTorch - several GB, be patient)'
$venv = Join-Path $Root '.venv'
if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
    & $pythonExe -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail 'could not create the virtualenv' }
    Write-Ok 'virtualenv created'
} else {
    Write-Ok 'virtualenv already exists'
}

$vpy = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $vpy)) { Fail 'cannot find the venv interpreter' }

& $vpy -m pip install --quiet --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Fail 'pip self-upgrade failed' }
Write-Ok 'pip up to date'

Write-Host '  installing clipsai ...'
& $vpy -m pip install --quiet 'clipsai==0.2.1'
if ($LASTEXITCODE -ne 0) { Fail 'clipsai install failed' }
Write-Ok 'clipsai installed'

Write-Host '  installing whisperx from git (slow) ...'
& $vpy -m pip install --quiet 'whisperx@git+https://github.com/m-bain/whisperx.git'
if ($LASTEXITCODE -ne 0) {
    Write-Warn 'whisperx install failed - transcription will not work until it succeeds'
    Write-Warn "  retry manually:  $vpy -m pip install `"whisperx@git+https://github.com/m-bain/whisperx.git`""
} else {
    Write-Ok 'whisperx installed'
}

# clipsai depends on python-magic, whose Windows build ships no libmagic DLL.
# This must run AFTER clipsai/whisperx or their dependency pulls overwrite it.
Write-Host '  installing python-magic-bin (Windows libmagic) ...'
& $vpy -m pip uninstall -y -q python-magic python-magic-bin 2>$null
& $vpy -m pip install --quiet python-magic-bin==0.4.14
if ($LASTEXITCODE -ne 0) { Write-Warn 'python-magic-bin failed - clipsai will not import' }
else { Write-Ok 'python-magic-bin installed (last, so it wins)' }

# clipsai calls nltk.download("punkt") at import time, but nltk >= 3.8.2 needs
# punkt_tab for sent_tokenize. Fetch both now so the first run is not a surprise.
Write-Host '  downloading nltk tokenizer data ...'
& $vpy -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
if ($LASTEXITCODE -ne 0) { Write-Warn 'nltk data download failed - transcription will fail at import' }
else { Write-Ok 'nltk punkt + punkt_tab ready' }

# --------------------------------------------------------------------- verify #
Write-Step 'Verifying imports'
$probe = @'
import importlib
# NB: not "clipsai" - its __init__ imports facenet-pytorch, which pins numpy<2
# while whisperx needs numpy>=2.1. The pipeline imports the submodules directly.
for name in ("clipsai.transcribe.transcriber", "clipsai.clip.clipfinder",
             "whisperx", "torch", "torchaudio", "magic", "nltk"):
    try:
        mod = importlib.import_module(name)
        print("  ok    %s %s" % (name, getattr(mod, "__version__", "")))
    except Exception as exc:
        print("  fail  %s -> %s: %s" % (name, type(exc).__name__, exc))
'@
$probeFile = Join-Path $env:TEMP 'clipsai_probe.py'
Set-Content -Path $probeFile -Value $probe -Encoding UTF8
& $vpy $probeFile
Remove-Item $probeFile -ErrorAction SilentlyContinue

Push-Location (Join-Path $Root 'yt-dlp')
& $vpy -c "import yt_dlp; print('  ok    yt-dlp ' + yt_dlp.version.__version__)"
Pop-Location

# ----------------------------------------------------------------- next steps #
Write-Step 'Done'
@"
Project root: $Root

Activate the environment:
  cd "$Root"
  .\.venv\Scripts\Activate.ps1

First run - plan only, nothing rendered, fast model:
  cd "$Root\yt-dlp"
  python shorts\make_shorts.py "<STREAM_OR_VOD_URL>" --dry-run --whisper-model base

Then render three clips with a static vertical crop:
  python shorts\make_shorts.py "<URL>" --whisper-model base --center-crop --max-clips 3

To enable speaker-tracking 9:16 reframing:
  1. create a token at https://huggingface.co/settings/tokens
  2. accept the terms on pyannote/speaker-diarization and pyannote/segmentation
  3. `$env:HUGGINGFACE_TOKEN = "hf_..."

To publish into Postiz:
  cd "$Root\postiz-app"
  docker compose up -d
  get an API key in Postiz: Settings -> Public API
  `$env:POSTIZ_API_KEY = "..."
  `$env:POSTIZ_API_URL = "http://localhost:5000/api"
  python shorts\make_shorts.py "<URL>" --postiz

Posts are created as drafts by default. Review them in Postiz before scheduling.
Full option list: yt-dlp\shorts\README.md
"@ | Write-Host
