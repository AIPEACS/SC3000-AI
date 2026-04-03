# run_task.ps1
# Runs the Exercise 1 Prolog task and captures trace output for the report.
#
# Usage (from Lab2/ or anywhere):
#   .\run_task.ps1
#   .\run_task.ps1 -Interactive   # opens swipl shell so you can step manually
#
# Output is saved to part1\trace_output.txt for use in your PDF report.

param(
    [switch]$Interactive
)

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$PlFile     = Join-Path $ScriptDir "part1\task.pl"
$OutFile    = Join-Path $ScriptDir "part1\trace_output.txt"

if (-not (Test-Path $PlFile)) {
    Write-Error "Could not find: $PlFile"
    exit 1
}

if ($Interactive) {
    Write-Host "Opening SWI-Prolog interactive shell..."
    Write-Host "  Inside, type:  trace.  then  unethical(stevey).  to step through."
    Write-Host ""
    swipl -s $PlFile
} else {
    Write-Host "Running Exercise 1: The Smart Phone Rivalry (auto trace)..."
    Write-Host "Saving output to: $OutFile"
    Write-Host ""

    # leash(-all) makes trace non-interactive (prints all steps without pausing)
    $output = swipl -s $PlFile -g "leash(-all), trace, unethical(stevey), notrace, halt." 2>&1

    $stripped = $output | ForEach-Object { "$_".TrimStart() }
    $stripped | Tee-Object -FilePath $OutFile

    Write-Host ""
    Write-Host "Done. Trace saved to: $OutFile"
    Write-Host "Copy the content into your PDF report."
}
