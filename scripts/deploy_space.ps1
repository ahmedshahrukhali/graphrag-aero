# Deploy the HF Space via HfApi HTTP commits (NOT git push).
#
# Deploy doctrine (PLAN_ZEROGPU): two prior PAUSES ("Flagged as abusive", S43/S44)
# were both caused by git-pushing the project tree to the Space remote — the mirror
# exposed local-dev tunnel tooling and a `--force` bypass re-armed the abuse scanner.
# New rules:
#   - No git remote to the Space (removed). No git client, no force push, ever.
#   - Stage a WHITELIST tree, then HfApi.upload_folder → one API commit.
#   - delete_patterns mirrors the whitelist (stale Space files are pruned).
#   - Any HF acknowledgement / terms error → resolve in the web UI, re-run this script.
#     Never escalate to force.
#
# Whitelist (S45 thin shell):
#   hf_space/space_root/*  -> staging ROOT   (README.md, app.py, requirements.txt)
#   hf_space/              -> staging/hf_space  (minus __pycache__ and space_root)
# NEVER staged: tunnel tooling (run_localtunnel.bat, wake_proxy.py, ecosystem.config.js,
#   scripts/tray), root Dockerfile, scripts/, data/, .env, other phases' tests.
#
# Usage:
#   $env:HF_TOKEN = "<write token>"; scripts\deploy_space.ps1
#   scripts\deploy_space.ps1 -DryRun      # stage + list the whitelist, no upload

param([switch]$DryRun)

$ErrorActionPreference = "Stop"

$spaceId = "ahmedsali/graphaero-rag"
$repo = (git rev-parse --show-toplevel).Trim()

if (-not $DryRun -and -not $env:HF_TOKEN) {
    throw "HF_TOKEN env var not set (write token required for the HfApi upload)."
}

$staging = Join-Path ([IO.Path]::GetTempPath()) ("space_deploy_" + [IO.Path]::GetRandomFileName())
$helper  = Join-Path ([IO.Path]::GetTempPath()) ("space_upload_" + [IO.Path]::GetRandomFileName() + ".py")
New-Item -ItemType Directory -Path $staging | Out-Null
try {
    # space_root/* -> staging root (README.md, app.py, requirements.txt)
    Copy-Item -Path (Join-Path $repo "hf_space\space_root\*") -Destination $staging -Recurse -Force

    # hf_space/ -> staging\hf_space, then strip __pycache__ and the nested space_root copy
    $dst = Join-Path $staging "hf_space"
    Copy-Item -Path (Join-Path $repo "hf_space") -Destination $dst -Recurse -Force
    Get-ChildItem -Path $dst -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    Remove-Item -Path (Join-Path $dst "space_root") -Recurse -Force -ErrorAction SilentlyContinue
    # The gradio SDK ignores Dockerfiles; never ship one (deploy doctrine).
    Remove-Item -Path (Join-Path $dst "Dockerfile") -Force -ErrorAction SilentlyContinue

    # Guard: assert no forbidden artifact slipped into the whitelist.
    $forbidden = @("run_localtunnel", "wake_proxy", "ecosystem.config", "\.env$",
                   "[/\\]Dockerfile$", "[/\\]tray[/\\]")
    $staged = Get-ChildItem -Path $staging -Recurse -File |
        ForEach-Object { $_.FullName.Substring($staging.Length + 1).Replace('\', '/') }
    foreach ($f in $staged) {
        foreach ($bad in $forbidden) {
            if ($f -match $bad) { throw "Whitelist violation: '$f' matches forbidden pattern '$bad'" }
        }
    }

    Write-Host "Staged $($staged.Count) files for $spaceId :"
    $staged | Sort-Object | ForEach-Object { Write-Host "  $_" }

    if ($DryRun) { Write-Host "`n[DryRun] No upload performed."; return }

    # Upload via HfApi (token from env; never embedded anywhere).
    @'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
info = api.upload_folder(
    repo_id="ahmedsali/graphaero-rag",
    repo_type="space",
    folder_path=sys.argv[1],
    commit_message="deploy: thin Gradio shell (S45) via HfApi",
    delete_patterns=["*"],
)
print("upload OK:", getattr(info, "commit_url", info))
'@ | Set-Content -Path $helper -Encoding utf8
    python $helper $staging
    if ($LASTEXITCODE -ne 0) { throw "HfApi upload failed (exit $LASTEXITCODE)." }
    Write-Host "Deployed thin shell to $spaceId."
} finally {
    Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
    Remove-Item -Force $helper -ErrorAction SilentlyContinue
}
