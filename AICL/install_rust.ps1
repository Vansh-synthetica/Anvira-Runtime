$rustup = "C:\Users\DELL\.cargo\bin\rustup.exe"
$out = "D:\rustup_install.log"
$err = "D:\rustup_install.err"
Write-Output "Starting rustup toolchain install at $(Get-Date)"
& $rustup toolchain install stable 2>$err >$out
Write-Output "Done at $(Get-Date). Exit code: $LASTEXITCODE"
