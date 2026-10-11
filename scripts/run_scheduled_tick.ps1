param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
$TaskRepo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $TaskRepo
if (Test-Path -LiteralPath (Join-Path $TaskRepo '.deps')) {
    $env:PYTHONPATH = (Join-Path $TaskRepo '.deps')
}
$TaskLogDirectory = Join-Path $TaskRepo 'data/prospective_automation/logs'
New-Item -ItemType Directory -Path $TaskLogDirectory -Force | Out-Null
$TaskLog = Join-Path $TaskLogDirectory ('scheduler-' + [DateTime]::UtcNow.ToString('yyyy-MM-dd') + '.log')
& $Python -m jobs.scheduled_tick >> $TaskLog 2>&1
exit $LASTEXITCODE
