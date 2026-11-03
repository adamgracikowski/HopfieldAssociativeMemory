<#
.SYNOPSIS
Downloads the images of a source list and verifies their SHA-1 checksums.

.DESCRIPTION
The source list is a tab-separated file with the columns file, sha1, url,
page, author and license. Each image is downloaded from `url` to
`<OutDir>/<file>`. Files that already exist with the expected checksum are
skipped, so the script can be run again safely. A checksum mismatch stops
the script and removes the partial download.

.EXAMPLE
pwsh -File data/images/download.ps1 -Sources data/images/cats/sources.tsv -OutDir data/images/cats/raw
#>
[CmdletBinding()]
param(
    # Tab-separated source list.
    [Parameter(Mandatory)]
    [string] $Sources,

    # Directory for the downloaded images.
    [Parameter(Mandatory)]
    [string] $OutDir
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$userAgent = 'HopfieldAssociativeMemory/0.1 (https://github.com/adamgracikowski/HopfieldAssociativeMemory)'

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$rows = Import-Csv -Path $Sources -Delimiter "`t" -Encoding UTF8

foreach ($row in $rows) {
    $target = Join-Path $OutDir $row.file
    if ((Test-Path $target) -and (Get-FileHash $target -Algorithm SHA1).Hash -eq $row.sha1) {
        Write-Host "ok          $($row.file)"
        continue
    }

    Write-Host "downloading $($row.file)"
    $partial = "$target.part"
    Invoke-WebRequest -Uri $row.url -OutFile $partial -UserAgent $userAgent
    $hash = (Get-FileHash $partial -Algorithm SHA1).Hash
    if ($hash -ne $row.sha1) {
        Remove-Item $partial
        throw "checksum mismatch for $($row.file): expected $($row.sha1), got $hash"
    }
    Move-Item -Force $partial $target
}

Write-Host "$($rows.Count) images in $OutDir"
