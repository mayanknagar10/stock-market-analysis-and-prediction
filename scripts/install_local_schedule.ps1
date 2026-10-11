param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
$TaskRepo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$TaskPython = (Resolve-Path -LiteralPath $Python).Path
$TaskScript = Join-Path $TaskRepo 'scripts/run_scheduled_tick.ps1'
$TaskName = 'StockPro-Prospective-Research-v1'
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw 'Task already exists; inspect it before changing its configuration.'
}
$TaskArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $TaskScript + '" -Python "' + $TaskPython + '"'
$TaskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $TaskArguments -WorkingDirectory $TaskRepo
$TaskTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
$TaskSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
$TaskPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $TaskAction -Trigger $TaskTrigger -Settings $TaskSettings -Principal $TaskPrincipal -Description 'Private research-only collection and outcome resolution. Requires this user logged in and persistent storage.' | Select-Object TaskName,State
