# Speaks bench/corpus.txt with the Windows voices into 16 kHz mono 16-bit WAV clips.
# Each clip gets a matching .txt reference transcript for the WER score in bench-asr.
param(
    [string]$OutDir = (Join-Path $PSScriptRoot "..\target\bench-audio")
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name })
if ($voices.Count -eq 0) { throw "No Windows voices are installed." }

$index = 0
foreach ($line in Get-Content (Join-Path $PSScriptRoot "corpus.txt") -Encoding UTF8) {
    if (-not $line.Trim()) { continue }
    $id, $text = $line.Split("|", 2)
    $synth.SelectVoice($voices[$index % $voices.Count])
    $wav = Join-Path $OutDir "$id.wav"
    $synth.SetOutputToWaveFile($wav, $format)
    $synth.Speak($text)
    $synth.SetOutputToNull()
    Set-Content -Path (Join-Path $OutDir "$id.txt") -Value $text -Encoding UTF8 -NoNewline
    $index++
}
$synth.Dispose()
Write-Output "wrote $index clips to $OutDir"
