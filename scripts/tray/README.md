# GraphRAG Aero — system-tray stack controller

AutoHotkey v2 tray icon that wraps the whole serving stack — the pm2 apps
(`wake-proxy` on :8080, `localtunnel` → `graphrag-aero-cocko.loca.lt`) **and**
the docker compose services — with no visible terminal. Tray on = stack on.

## Files

| File | Purpose |
|---|---|
| `graphrag_tray.ahk` | The tray app (AutoHotkey v2). |
| `make_icons.ps1` | One-time generator for `icons/*.ico` (gitignored, regenerate anywhere). |

## One-time setup

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\tray\make_icons.ps1
```

## Usage

Double-click `graphrag_tray.ahk` (AHK v2 is associated with `.ahk`), or:

```powershell
& "C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe" scripts\tray\graphrag_tray.ahk
```

On launch it **auto-starts the stack** (`docker compose up -d` + `pm2 start
ecosystem.config.js`) unless already up. Pass `--no-start` to only monitor.

Icon colors (polled every 15 s):

- 🟢 green — pm2 (both apps) + docker containers running
- 🟠 amber — partial, e.g. docker up but pm2/tunnel down ⇒ HF Space gets 503
- ⚪ gray — everything stopped
- 🔵 blue — starting / stopping

Menu (right-click; **double-click toggles start/stop**):

- Start / Stop / Toggle stack
- Open UI (`localhost:7860`) · Open tunnel `/healthz`
- **Exit (stop stack)** — pm2 stop + `docker compose stop`, then quit
- Exit (leave stack running) — quit the tray only

## Run at login

`Win+R` → `shell:startup` → create a shortcut with target:

```
"C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe" "C:\Users\cocko\workspace\graphrag-aero\scripts\tray\graphrag_tray.ahk"
```

(Docker Desktop must itself be running/autostarted for the compose half;
the tray starts *containers*, not the Docker engine.)

## Troubleshooting

- Last polled state is written to `%TEMP%\graphrag_tray_status.txt`
  (e.g. `partial | pm2 down · docker 7 running`).
- pm2 details: `pm2 list`, `pm2 logs localtunnel --lines 30 --nostream`.
- Note `wake-proxy` independently stops the containers after 30 min idle —
  the icon turning amber while the tunnel stays up is that auto-sleep, not a
  failure; the next request through the tunnel wakes them.
