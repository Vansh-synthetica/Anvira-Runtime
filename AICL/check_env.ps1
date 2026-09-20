# Check Rust
$rustPaths = Get-ChildItem "C:\Users\*\.cargo\bin\cargo.exe" -ErrorAction SilentlyContinue
if ($rustPaths) {
    $rustPath = $rustPaths[0].FullName
    Write-Output "Rust found: $rustPath"
    & $rustPath --version
} else {
    Write-Output "Rust NOT found"
}

# Check C++ compilers
$gcc = Get-Command gcc -ErrorAction SilentlyContinue
if ($gcc) { Write-Output "GCC found: $($gcc.Source)" } else { Write-Output "GCC not found" }

$clang = Get-Command clang -ErrorAction SilentlyContinue
if ($clang) { Write-Output "Clang found: $($clang.Source)" } else { Write-Output "Clang not found" }

$msvc = Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC" -ErrorAction SilentlyContinue
if ($msvc) { Write-Output "MSVC found: $($msvc.FullName)" } else { Write-Output "MSVC not found" }

# Check for clang-cl
$clangcl = Get-Command "clang-cl" -ErrorAction SilentlyContinue
if ($clangcl) { Write-Output "clang-cl found: $($clangcl.Source)" } else { Write-Output "clang-cl not found" }

# Check for ninja
$ninja = Get-Command ninja -ErrorAction SilentlyContinue
if ($ninja) { Write-Output "Ninja found: $($ninja.Source)" } else { Write-Output "Ninja not found" }

# List installed VS workloads
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if (Test-Path $vsPath) {
    $componentsFile = Join-Path $vsPath "instance\instance.lock.biao"
    $componentsFile2 = Join-Path $vsPath "instance\instance.lock"
    if (Test-Path $componentsFile) { Write-Output "VS instance file found" }
    if (Test-Path $componentsFile2) { Write-Output "VS instance file 2 found" }
}
