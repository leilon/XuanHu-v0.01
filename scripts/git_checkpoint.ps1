param(
  [string]$Message = ""
)

$ErrorActionPreference = "Stop"
if ($Message -eq "") {
  $Message = "checkpoint: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

git add -A
git commit -m $Message
Write-Host "[ok] commit created: $Message"

