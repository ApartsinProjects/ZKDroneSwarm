param(
    [Parameter(Mandatory=$true)][string]$In,
    [Parameter(Mandatory=$true)][string]$Out
)
# Convert a .docx to .pdf using the installed Microsoft Word via COM automation.
# wdFormatPDF = 17
$ErrorActionPreference = "Stop"
$inAbs  = (Resolve-Path $In).Path
$outAbs = [System.IO.Path]::GetFullPath($Out)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($inAbs, $false, $true)  # ConfirmConversions, ReadOnly
    # Update fields (page numbers in footer) before export
    $doc.Fields.Update() | Out-Null
    $doc.SaveAs([ref]$outAbs, [ref]17)
    $doc.Close([ref]$false)
    Write-Output "PDF_OK $outAbs"
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
