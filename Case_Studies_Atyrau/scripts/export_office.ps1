param([string]$Case='all',[switch]$ReportsOnly,[switch]$SlidesOnly)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$dirs=Get-ChildItem -LiteralPath $root -Directory | Where-Object { $_.Name -match '^0[1-4]_' -and ($Case -eq 'all' -or ($Case -split ',') -contains $_.Name.Substring(0,2)) }
if(-not $SlidesOnly){
  $word=New-Object -ComObject Word.Application
  Write-Output 'Word converter started'
  $word.Visible=$false
  $word.DisplayAlerts=0
  try {
    foreach($d in $dirs){
      $artifactPath=Join-Path $d.FullName 'Report.docx'
      if(Test-Path -LiteralPath $artifactPath){
        Write-Output ('Opening '+$artifactPath)
        $doc=$word.Documents.Open($artifactPath,$false,$true)
        Write-Output 'Opened report'
        try {
          $pdf=Join-Path $d.FullName 'Report.pdf'
          $exportDir=Join-Path (Split-Path -Parent $root) '_build\office_exports'
          New-Item -ItemType Directory -Path $exportDir -Force | Out-Null
          $temporaryPdf=Join-Path $exportDir ($d.Name+'-'+[guid]::NewGuid().ToString()+'.pdf')
          Write-Output 'Exporting report PDF'
          $doc.ExportAsFixedFormat($temporaryPdf,17)
          Copy-Item -LiteralPath $temporaryPdf -Destination $pdf -Force
          Write-Output "$($d.Name) report exported"
        } finally { $doc.Close(0) }
      }
    }
  } finally { $word.Quit() }
}
if(-not $ReportsOnly){
  $powerpoint=New-Object -ComObject PowerPoint.Application
  $powerpoint.DisplayAlerts=1
  try {
    foreach($d in $dirs){
      $artifactPath=Join-Path $d.FullName 'Presentation.pptx'
      if(Test-Path -LiteralPath $artifactPath){
        $deck=$powerpoint.Presentations.Open($artifactPath,$true,$false,$false)
        try {
          $pdf=Join-Path $d.FullName 'Presentation.pdf'
          $deck.SaveAs($pdf,32)
          Write-Output "$($d.Name) slides=$($deck.Slides.Count)"
        } finally { $deck.Close(); [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck); $deck=$null }
      }
    }
  } finally { $powerpoint.Quit(); [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerpoint); $powerpoint=$null; [GC]::Collect(); [GC]::WaitForPendingFinalizers() }
}
