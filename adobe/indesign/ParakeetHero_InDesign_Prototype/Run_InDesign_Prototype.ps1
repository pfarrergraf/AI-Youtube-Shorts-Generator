$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$jsxPath = Join-Path $projectRoot 'ParakeetHero_BUILD.jsx'
$outputPath = Join-Path $projectRoot 'output_final'

$hadExistingInDesign = [bool](Get-Process -Name InDesign -ErrorAction SilentlyContinue)
if (-not (Test-Path -LiteralPath $jsxPath)) {
    throw "Build script not found: $jsxPath"
}

$app = $null
try {
    $app = New-Object -ComObject 'InDesign.Application.2026'
    if ($hadExistingInDesign) {
        foreach ($document in @($app.Documents)) {
            $documentPath = ''
            try { $documentPath = [string]$document.FullName } catch {}
            if ($documentPath -and -not $documentPath.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "InDesign has an unrelated document open; refusing automation: $documentPath"
            }
        }
    }
    $app.DoScript($jsxPath, 1246973031)
} finally {
    if ($null -ne $app) {
        if (-not $hadExistingInDesign) {
            try { $app.Quit() } catch {}
        }
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($app) | Out-Null
    }
}

$required = @(
    'ParakeetHero_9x16.indd',
    'ParakeetHero_9x16.idml',
    'ParakeetHero_9x16.png',
    'ParakeetHero_16x9.indd',
    'ParakeetHero_16x9.idml',
    'ParakeetHero_16x9.png',
    'build.log'
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $outputPath $_)) })
if ($missing.Count -gt 0) {
    throw "InDesign finished without required outputs: $($missing -join ', ')"
}

Get-ChildItem -LiteralPath $outputPath -File | Sort-Object Name | Select-Object Name, Length, LastWriteTime
