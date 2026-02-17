# save as: dev_rebuild_touchup.ps1 (at repo root)
param(
  [switch]$Clean,
  [switch]$RunApp
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location "$repo\tools\midi_touchup_editor_rust"
if ($Clean) { cargo clean }
cargo build --release
Pop-Location

if ($RunApp) {
  & "$repo\.venv-win\Scripts\python.exe" "$repo\synthesia2midi\run.py"
}
