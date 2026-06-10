# Deploy the MINIMAL Space tree to the HF Space repo.
#
# Pushes a single orphan commit containing ONLY what the Space build consumes
# (root Dockerfile does `COPY hf_space /app/hf_space`):
#   README.md  Dockerfile  hf_space/
#
# Why not a full mirror: HF's abuse scanner PAUSED the Space ("Flagged as
# abusive", S43) when the mirror exposed local-dev tunnel tooling
# (run_localtunnel.bat, cloudflared/localtunnel refs in ecosystem.config.js,
# scripts/tray). None of it is needed — or wanted — on the Space.
#
# ALL Space deploys go through this script now. Do NOT `git push huggingface
# main` from the project repo — that recreates the full mirror and re-arms
# the flag. Local main intentionally diverges from huggingface/main.

$ErrorActionPreference = "Stop"

$repo = git rev-parse --show-toplevel
$remoteUrl = git -C $repo remote get-url huggingface
$sha = git -C $repo rev-parse --short HEAD

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("space_deploy_" + [IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    # Tracked content only (no __pycache__, no scratch files).
    $tar = Join-Path $tmp "tree.tar"
    git -C $repo archive --output $tar HEAD README.md Dockerfile hf_space
    tar -xf $tar -C $tmp
    Remove-Item $tar

    git -C $tmp init -b main | Out-Null
    git -C $tmp add -A
    git -C $tmp -c user.name="deploy" -c user.email="deploy@local" `
        commit -m "deploy: hf_space @ $sha (minimal tree)" | Out-Null
    git -C $tmp push --force $remoteUrl main

    Write-Host "Deployed minimal tree @ $sha to the Space."
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
