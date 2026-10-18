[CmdletBinding()]
param(
    [switch]$SkipGitHub
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    foreach ($tool in @('git', 'curl.exe')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Required tool is missing: $tool"
        }
    }
    if (-not $SkipGitHub -and -not (Get-Command gh -ErrorAction SilentlyContinue)) {
        throw 'GitHub CLI is missing. Install gh, then rerun this script.'
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw 'Install Visual Studio Build Tools with Desktop development with C++, then rerun.'
    }
    $cppInstall = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $cppInstall) {
        throw 'The Visual Studio C++ toolchain is missing. Add the C++ workload and Windows SDK.'
    }

    $cargoRoot = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
    $cargoBin = Join-Path $cargoRoot 'bin'
    $env:Path = "$cargoBin;$env:Path"

    if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
        $setupDir = Join-Path $projectRoot '.setup'
        New-Item -ItemType Directory -Path $setupDir -Force | Out-Null
        $installer = Join-Path $setupDir 'rustup-init.exe'
        $checksumFile = Join-Path $setupDir 'rustup-init.exe.sha256'
        $downloadUrl = 'https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe'
        Invoke-Checked -Executable 'curl.exe' -Arguments @('--fail', '--location', '--connect-timeout', '15', '--max-time', '300', '--output', $installer, $downloadUrl)
        Invoke-Checked -Executable 'curl.exe' -Arguments @('--fail', '--location', '--connect-timeout', '15', '--max-time', '60', '--output', $checksumFile, "$downloadUrl.sha256")
        $expectedHash = ((Get-Content -LiteralPath $checksumFile -Raw).Trim() -split '\s+')[0]
        $actualHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
        if ($expectedHash -notmatch '^[0-9a-fA-F]{64}$' -or $actualHash -ne $expectedHash) {
            throw 'Rust installer checksum mismatch. Installation stopped.'
        }
        Invoke-Checked -Executable $installer -Arguments @('-y', '--profile', 'minimal', '--default-toolchain', 'stable', '--component', 'rustfmt', '--component', 'clippy')
    }

    Invoke-Checked -Executable 'rustup' -Arguments @('toolchain', 'install', 'stable', '--profile', 'minimal', '--component', 'rustfmt', '--component', 'clippy')
    Invoke-Checked -Executable 'rustc' -Arguments @('--version')
    Invoke-Checked -Executable 'cargo' -Arguments @('--version')
    Invoke-Checked -Executable 'cargo' -Arguments @('fetch')
    Invoke-Checked -Executable 'cargo' -Arguments @('fmt', '--all', '--', '--check')
    Invoke-Checked -Executable 'cargo' -Arguments @('clippy', '--all-targets', '--', '-D', 'warnings')
    Invoke-Checked -Executable 'cargo' -Arguments @('test', '--all-targets')
    Invoke-Checked -Executable 'cargo' -Arguments @('run', '--', 'info')

    if (-not $SkipGitHub) {
        & gh auth status
        if ($LASTEXITCODE -ne 0) {
            throw 'GitHub authentication is unavailable. Check network access, run gh auth login --hostname github.com --web if needed, then rerun setup.'
        }

        $remotes = @(& git remote)
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not inspect Git remotes.'
        }
        if ($remotes -contains 'origin') {
            $existingRemote = & git remote get-url origin
            if ($LASTEXITCODE -ne 0) {
                throw 'Could not read the origin remote.'
            }
            Write-Host "Origin already exists: $existingRemote"
            Write-Host 'Build verification succeeded. Existing remote left unchanged.'
        } else {
            Invoke-Checked -Executable 'git' -Arguments @('add', '--', 'Cargo.lock')
            & git diff --cached --quiet -- Cargo.lock
            $stagedStatus = $LASTEXITCODE
            if ($stagedStatus -eq 1) {
                Invoke-Checked -Executable 'git' -Arguments @('commit', '--only', '-m', 'Lock Rust dependencies after build verification', '--', 'Cargo.lock')
            } elseif ($stagedStatus -ne 0) {
                throw 'Could not inspect staged changes.'
            }
            Invoke-Checked -Executable 'gh' -Arguments @('repo', 'create', 'fast-tokenizer', '--private', '--description', 'Rust tokenization and document preparation engine', '--source', '.', '--remote', 'origin', '--push')
        }
    }

    Write-Host 'Prerequisite setup and build verification completed.'
} finally {
    Pop-Location
}
