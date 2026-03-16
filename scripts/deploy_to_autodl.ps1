param(
  [Parameter(Mandatory = $true)][string]$User,
  [Parameter(Mandatory = $true)][string]$Host,
  [Parameter(Mandatory = $true)][int]$Port,
  [Parameter(Mandatory = $true)][string]$RemoteDir,
  [string]$IdentityFile = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$ArchivePath = Join-Path $env:TEMP "agentic_medical_gpt.tar.gz"

if (Test-Path $ArchivePath) {
  Remove-Item $ArchivePath -Force
}

Write-Host "[step] pack project"
tar -czf $ArchivePath `
  --exclude=.git `
  --exclude=__pycache__ `
  --exclude=.venv `
  -C $ProjectRoot .

$sshBase = "ssh -p $Port"
$scpBase = "scp -P $Port"
if ($IdentityFile -ne "") {
  $sshBase = "$sshBase -i `"$IdentityFile`""
  $scpBase = "$scpBase -i `"$IdentityFile`""
}

Write-Host "[step] create remote dir"
Invoke-Expression "$sshBase $User@$Host `"mkdir -p '$RemoteDir'`""

Write-Host "[step] upload archive"
Invoke-Expression "$scpBase `"$ArchivePath`" $User@$Host:`"$RemoteDir/project.tar.gz`""

Write-Host "[step] unpack and bootstrap"
Invoke-Expression "$sshBase $User@$Host `"cd '$RemoteDir' && tar -xzf project.tar.gz && bash scripts/autodl_remote_bootstrap.sh '$RemoteDir'`""

Write-Host "[step] run smoke+benchmark"
Invoke-Expression "$sshBase $User@$Host `"bash '$RemoteDir/scripts/autodl_remote_run.sh' '$RemoteDir'`""

Write-Host "[ok] deployed and executed"

