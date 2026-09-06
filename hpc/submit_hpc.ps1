# Run from PowerShell: .\hpc\submit_hpc.ps1
# Updates the HPC checkout and submits the job there through Slurm.
param(
    [ValidatePattern('^[A-Za-z0-9_-]+\.sh$')]
    [string]$Script = 'test_shell.sh'
)
$ErrorActionPreference = 'Stop'

$keyPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'server_access/private_key'
$knownHostsPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'server_access/known_hosts'
if (-not (Test-Path -LiteralPath $keyPath -PathType Leaf)) {
    throw "SSH private key not found: $keyPath"
}

$sshPath = Join-Path $env:WINDIR 'System32/OpenSSH/ssh.exe'
if (-not (Test-Path -LiteralPath $sshPath)) {
    throw 'Windows OpenSSH Client is required.'
}

# && prevents submission if changing directory or pulling fails.
# SSH exits when the remote command finishes, including on failure.
# Disable connection sharing/persistence so no SSH connection stays open.
# The submitted Slurm job continues independently after disconnection.
function Invoke-HpcCommand([string]$Command) {
& $sshPath -T -i $keyPath -o IdentitiesOnly=yes -o BatchMode=yes `
    -o ControlMaster=no -o ControlPath=none -o ControlPersist=no `
    -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$knownHostsPath" `
    -o ConnectTimeout=20 'alex116u1@login02.c2.hpc.bibalex.org' $Command
if ($LASTEXITCODE -ne 0) {
    throw "HPC update/submission failed (SSH exit code $LASTEXITCODE)."
}
}
$stem = [IO.Path]::GetFileNameWithoutExtension($Script)
$remoteCommand = 'cd ~/data/VirtualSink && git pull && cd hpc && sbatch --parsable --chdir="$PWD" --output="$PWD/{0}-%j.out" --error="$PWD/{0}-%j.err" {1}' -f $stem, $Script
$submission = @(Invoke-HpcCommand $remoteCommand)
$submission | ForEach-Object { Write-Host $_ }
$jobLine = $submission | Where-Object { $_ -match '^\d+(;[^\s]+)?$' } | Select-Object -Last 1
if (-not $jobLine) { throw 'Cannot identify submitted job ID. Check Slurm before retrying to avoid a duplicate.' }
$jobId = ($jobLine -split ';')[0]
Write-Host "Submitted job $jobId. SSH closed. Waiting 20 seconds before checking output."
Start-Sleep -Seconds 20
$checkCommand = 'squeue -j {0}; sacct -j {0} --format=JobID,State,ExitCode; if test -f "$HOME/data/VirtualSink/hpc/{1}-{0}.out"; then tail -n 80 "$HOME/data/VirtualSink/hpc/{1}-{0}.out"; else echo "OUTPUT NOT CREATED: execution is not confirmed; inspect queue state and check again."; fi; if test -f "$HOME/data/VirtualSink/hpc/{1}-{0}.err"; then tail -n 80 "$HOME/data/VirtualSink/hpc/{1}-{0}.err"; fi' -f $jobId, $stem
Invoke-HpcCommand $checkCommand
Write-Host 'Output check finished; SSH closed. Review status and logs before claiming execution or success.'
