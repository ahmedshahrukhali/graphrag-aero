---
title: GraphRAG Aero
emoji: 🛩️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.12"
pinned: false
---

# GraphRAG Aero

Cross-lingual (EN + FR) GraphRAG over **Transport Canada Advisory Circulars** and
**TSB aviation investigation reports**, with cited PDF pages rendered server-side
as the multimodal output.

This Space is the **Gradio shell**. It streams each query from a FastAPI backend
(`BACKEND_URL` secret) and renders the cited source pages inline. A self-contained
**ZeroGPU** engine + toggle (run the full pipeline in-Space, fall back to the
backend on quota exhaustion) lands in a later stage.

## Deploy

This Space is deployed by **HfApi HTTP commits** from the project repo — never by
`git push` to the Space remote (two prior pauses were caused by mirror pushes
exposing local-dev tooling). From the repo root:

```powershell
$env:HF_TOKEN = "<write token>"   # write scope
scripts/deploy_space.ps1          # stages the whitelist tree, then HfApi.upload_folder
```

`scripts/deploy_space.ps1 -DryRun` lists the staged whitelist without uploading.

## Configure

Set `BACKEND_URL` as a Space secret pointing at a publicly reachable backend
(e.g. the project's localtunnel `https://graphrag-aero-cocko.loca.lt`). Without it
the Space boots but chat cannot reach the backend.
