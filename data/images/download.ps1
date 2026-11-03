<#
.SYNOPSIS
Downloads the images of a source list and verifies their SHA-1 checksums.

.DESCRIPTION
The source list is a tab-separated file with the columns file, sha1, url,
page, author, license and license_url. Each image is downloaded from `url` to
`<OutDir>/<file>`, where `file` must be a plain file name. Files that already
exist with the expected checksum are skipped, so the script can be run again
safely. A checksum mismatch stops the script and removes the partial
download.

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
$rows = @(Import-Csv -LiteralPath $Sources -Delimiter "`t" -Encoding UTF8)

foreach ($row in $rows) {
    if ([System.IO.Path]::GetFileName($row.file) -ne $row.file) {
        throw "image $($row.file) is not a plain file name"
    }
    $target = Join-Path $OutDir $row.file
    if ((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target -Algorithm SHA1).Hash -eq $row.sha1) {
        Write-Host "ok          $($row.file)"
        continue
    }

    Write-Host "downloading $($row.file)"
    $partial = "$target.part"
    Invoke-WebRequest -Uri $row.url -OutFile $partial -UserAgent $userAgent
    $hash = (Get-FileHash -LiteralPath $partial -Algorithm SHA1).Hash
    if ($hash -ne $row.sha1) {
        Remove-Item -LiteralPath $partial
        throw "checksum mismatch for $($row.file): expected $($row.sha1), got $hash"
    }
    Move-Item -Force -LiteralPath $partial -Destination $target
}

Write-Host "$($rows.Count) images in $OutDir"
