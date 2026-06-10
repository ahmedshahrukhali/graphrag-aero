# Generates the tray-state .ico files used by graphrag_tray.ahk.
# Pure local file generation (System.Drawing) - no network/Docker calls.

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class GraphragTrayIconNative {
    [DllImport("user32.dll")]
    public static extern bool DestroyIcon(IntPtr hIcon);
}
"@

$iconDir = Join-Path $PSScriptRoot "icons"
New-Item -ItemType Directory -Force -Path $iconDir | Out-Null

function New-DotIcon {
    param(
        [string]$Path,
        [System.Drawing.Color]$Color
    )

    $bmp = [System.Drawing.Bitmap]::new(32, 32)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear([System.Drawing.Color]::Transparent)

    $brush = [System.Drawing.SolidBrush]::new($Color)
    $pen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(70, 70, 70), 1.5)
    $g.FillEllipse($brush, 2, 2, 28, 28)
    $g.DrawEllipse($pen, 2, 2, 28, 28)

    $hIcon = $bmp.GetHicon()
    $icon = [System.Drawing.Icon]::FromHandle($hIcon)
    $fs = [System.IO.File]::Create($Path)
    $icon.Save($fs)
    $fs.Close()

    $icon.Dispose()
    [GraphragTrayIconNative]::DestroyIcon($hIcon) | Out-Null
    $brush.Dispose()
    $pen.Dispose()
    $g.Dispose()
    $bmp.Dispose()
}

# green  = stack fully up (pm2 + docker compose)
New-DotIcon (Join-Path $iconDir "on.ico")      ([System.Drawing.Color]::FromArgb(46, 160, 67))
# gray   = stack fully down
New-DotIcon (Join-Path $iconDir "off.ico")     ([System.Drawing.Color]::FromArgb(140, 140, 140))
# amber  = mixed (e.g. docker up but pm2/tunnel down)
New-DotIcon (Join-Path $iconDir "partial.ico") ([System.Drawing.Color]::FromArgb(230, 160, 20))
# blue   = transition in progress (starting/stopping)
New-DotIcon (Join-Path $iconDir "busy.ico")    ([System.Drawing.Color]::FromArgb(70, 130, 220))

Write-Host "Icons written to $iconDir"
