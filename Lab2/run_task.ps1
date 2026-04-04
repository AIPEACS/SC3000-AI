# run_task.ps1
# Runs Lab 2 Prolog tasks and captures trace output for the report.
#
# Usage (from Lab2/):
#   .\run_task.ps1                # prompts for part selection
#   .\run_task.ps1 -Interactive   # opens swipl shell for manual stepping

param(
    [switch]$Interactive
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- Part selection ---
Write-Host "Select part to run:"
Write-Host "  1 - Exercise 1: Smart Phone Rivalry"
Write-Host "  2 - Exercise 2: Royal Family Succession"
$choice = Read-Host "Enter 1 or 2"

if ($choice -eq "1") {

    $PlFile  = Join-Path $ScriptDir "part1\task.pl"
    $OutFile = Join-Path $ScriptDir "part1\trace_output.txt"

    if (-not (Test-Path $PlFile)) { Write-Error "Could not find: $PlFile"; exit 1 }

    if ($Interactive) {
        Write-Host "Opening SWI-Prolog interactive shell..."
        Write-Host "  Inside, type:  trace.  then  unethical(stevey)."
        Write-Host ""
        swipl -s $PlFile
    } else {
        Write-Host ""
        Write-Host "Running Exercise 1: The Smart Phone Rivalry..."
        Write-Host "Saving output to: $OutFile"
        Write-Host ""

        $output  = swipl -s $PlFile -g "leash(-all), trace, unethical(stevey), notrace, halt." 2>&1
        $stripped = $output | ForEach-Object { "$_".TrimStart() }
        $stripped | Tee-Object -FilePath $OutFile

        Write-Host ""
        Write-Host "Done. Trace saved to: $OutFile"
    }

} elseif ($choice -eq "2") {

    $Pl21    = Join-Path $ScriptDir "part2\task2_1.pl"
    $Pl22    = Join-Path $ScriptDir "part2\task2_2.pl"
    $Out21   = Join-Path $ScriptDir "part2\task2_1_LOS.txt"
    $Out22   = Join-Path $ScriptDir "part2\task2_2_LOS.txt"

    if (-not (Test-Path $Pl21)) { Write-Error "Could not find: $Pl21"; exit 1 }
    if (-not (Test-Path $Pl22)) { Write-Error "Could not find: $Pl22"; exit 1 }

    if ($Interactive) {
        Write-Host "Select sub-task:"
        Write-Host "  1 - Old succession rule (task2_1.pl)"
        Write-Host "  2 - New succession rule (task2_2.pl)"
        $sub = Read-Host "Enter 1 or 2"
        if ($sub -eq "1") { swipl -s $Pl21 }
        else               { swipl -s $Pl22 }
    } else {
        # --- 2.1: Old rule ---
        Write-Host ""
        Write-Host "Running Exercise 2.1: Old Succession Rule (males first, then females)..."
        Write-Host "Saving output to: $Out21"
        Write-Host ""

        $out = swipl -s $Pl21 -g "leash(-all), trace, line_of_succession(elizabeth, X), notrace, halt." 2>&1
        $out | ForEach-Object { "$_".TrimStart() } | Tee-Object -FilePath $Out21

        # --- 2.2: New rule ---
        Write-Host ""
        Write-Host "Running Exercise 2.2: New Succession Rule (birth order, gender-independent)..."
        Write-Host "Saving output to: $Out22"
        Write-Host ""

        $out = swipl -s $Pl22 -g "leash(-all), trace, line_of_succession(elizabeth, X), notrace, halt." 2>&1
        $out | ForEach-Object { "$_".TrimStart() } | Tee-Object -FilePath $Out22

        Write-Host ""
        Write-Host "Done. Traces saved to part2\ for your report."
    }

} else {
    Write-Error "Invalid choice. Please enter 1 or 2."
    exit 1
}
