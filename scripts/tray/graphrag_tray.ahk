#Requires AutoHotkey v2.0
#SingleInstance Force
; GraphRAG Aero stack tray controller.
;
; Wraps the pm2 apps (wake-proxy + localtunnel) and the docker compose stack
; behind a system-tray icon. No console window is ever shown — everything runs
; hidden. Exiting via the tray stops the stack (tray on = stack on).
;
;   green  = pm2 + docker both up        gray = both down
;   amber  = partial (e.g. docker up, pm2 down)
;   blue   = starting / stopping
;
; Launch flags:
;   --no-start   monitor only; do not auto-start the stack on launch

Persistent()

; ---------- paths / config ----------
SplitPath(A_ScriptDir, , &scriptsDir)        ; ...\scripts
SplitPath(scriptsDir, , &repoRoot)           ; repo root
global REPO        := repoRoot
global PM2_APPS    := "wake-proxy localtunnel"
global UI_URL      := "http://localhost:7860"
global TUNNEL_URL  := "https://graphrag-aero-cocko.loca.lt/healthz"
global ICON_DIR    := A_ScriptDir "\icons"
global STATUS_FILE := A_Temp "\graphrag_tray_status.txt"
global POLL_MS     := 15000

global gBusy := false
global gPm2Up := false, gDockerN := 0

if !FileExist(ICON_DIR "\on.ico") {
    MsgBox("Icons missing — run scripts\tray\make_icons.ps1 first.`n`n"
         . "Continuing with the default AutoHotkey icon.", "GraphRAG tray", "Iconi")
}

; ---------- tray menu ----------
A_TrayMenu.Delete()
A_TrayMenu.Add("Toggle stack", ToggleStack)
A_TrayMenu.Add()
A_TrayMenu.Add("Start stack", (*) => StartStack())
A_TrayMenu.Add("Stop stack", (*) => StopStack())
A_TrayMenu.Add()
A_TrayMenu.Add("Open UI (localhost:7860)", (*) => Run(UI_URL))
A_TrayMenu.Add("Open tunnel /healthz", (*) => Run(TUNNEL_URL))
A_TrayMenu.Add("Refresh status", (*) => PollStatus())
A_TrayMenu.Add()
A_TrayMenu.Add("Exit (stop stack)", ExitStopStack)
A_TrayMenu.Add("Exit (leave stack running)", (*) => ExitApp())
A_TrayMenu.SetDefault("Toggle stack")        ; double-click toggles

SetIconState("busy", "starting…")
PollStatus()
if !HasArg("--no-start") && !(gPm2Up && gDockerN > 0)
    StartStack()
SetTimer(PollStatus, POLL_MS)

; ---------- actions ----------
StartStack(*) {
    global gBusy
    if gBusy
        return
    gBusy := true
    SetIconState("busy", "starting stack…")
    RunHidden("docker compose up -d", REPO)
    RunHidden("pm2 start ecosystem.config.js", REPO)
    gBusy := false
    PollStatus()
}

StopStack(*) {
    global gBusy
    if gBusy
        return
    gBusy := true
    SetIconState("busy", "stopping stack…")
    RunHidden("pm2 stop " PM2_APPS, REPO)
    RunHidden("docker compose stop", REPO)
    gBusy := false
    PollStatus()
}

ToggleStack(*) {
    global gPm2Up, gDockerN
    if (gPm2Up && gDockerN > 0)
        StopStack()
    else
        StartStack()
}

ExitStopStack(*) {
    StopStack()
    ExitApp()
}

; ---------- status ----------
PollStatus(*) {
    global gBusy, gPm2Up, gDockerN
    if gBusy
        return

    ; pm2: `pm2 pid <app>` prints the pid, or 0 when stopped
    out := RunCapture("(pm2 pid wake-proxy & pm2 pid localtunnel)")
    alive := 0
    for line in StrSplit(out, "`n", "`r") {
        line := Trim(line)
        if (line != "" && IsInteger(line) && Integer(line) > 0)
            alive += 1
    }
    gPm2Up := (alive = 2)

    ; docker: count running compose containers (read-only)
    out := RunCapture('docker compose ps --status running --format "{{.Name}}"', REPO)
    n := 0
    for line in StrSplit(out, "`n", "`r")
        if InStr(line, "graphrag")
            n += 1
    gDockerN := n

    pm2Txt := gPm2Up ? "up" : "down"
    if (gPm2Up && gDockerN > 0)
        SetIconState("on", "pm2 " pm2Txt " · docker " gDockerN " running")
    else if (!gPm2Up && gDockerN = 0)
        SetIconState("off", "stack stopped")
    else
        SetIconState("partial", "pm2 " pm2Txt " · docker " gDockerN " running")
}

SetIconState(state, detail) {
    icon := ICON_DIR "\" state ".ico"
    if FileExist(icon)
        TraySetIcon(icon, , true)
    A_IconTip := "GraphRAG Aero — " state "`n" detail
    try {
        f := FileOpen(STATUS_FILE, "w")
        f.Write(state " | " detail)
        f.Close()
    }
}

; ---------- helpers ----------
RunHidden(cmdline, workDir := "") {
    try RunWait(A_ComSpec " /c " cmdline, workDir, "Hide")
}

RunCapture(cmdline, workDir := "") {
    tmp := A_Temp "\grt_" A_TickCount "_" Random(1000, 9999) ".txt"
    try RunWait(A_ComSpec " /c " cmdline " > `"" tmp "`" 2>&1", workDir, "Hide")
    out := ""
    try out := FileRead(tmp)
    try FileDelete(tmp)
    return out
}

HasArg(name) {
    for a in A_Args
        if (a = name)
            return true
    return false
}
