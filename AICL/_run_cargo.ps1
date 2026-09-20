$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
$ErrorActionPreference = "Continue"
$ucrtLib = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64"
$umLib   = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64"
$env:LIB = "$ucrtLib;$umLib"
$env:LIBPATH = "$ucrtLib;$umLib"
Set-Location "D:\New folder (6)\Projects\AICL\core-rust"
$out = & rustc --edition 2021 --emit=metadata src/lib.rs -o NUL 2>&1
$lines = $out | Out-String
$errors = ($lines | Select-String -Pattern "^error").Count
$warnings = ($lines | Select-String -Pattern "^warning").Count
Write-Host "=== Total errors: $errors ==="
Write-Host "=== Total warnings: $warnings ==="
$out | Out-String | Write-Host
Write-Host "EXITCODE: $LASTEXITCODE"
