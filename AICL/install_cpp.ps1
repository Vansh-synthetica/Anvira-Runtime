$setup = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe"
$log = "D:\cpp_install.log"
$err = "D:\cpp_install.err"
Write-Output "Starting VS modify at $(Get-Date)"

# BuildTools C++ workload id: Microsoft.VisualStudio.Workload.VCTools
# Components: Microsoft.VisualStudio.Component.VC.Tools.x86.x64, Microsoft.VisualStudio.Component.Windows10SDK
$argList = @(
    "modify"
    "--installPath", "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    "--add", "Microsoft.VisualStudio.Workload.VCTools"
    "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
    "--add", "Microsoft.VisualStudio.Component.Windows10SDK.20348"
    "--quiet"
    "--norestart"
    "--noprogress"
    "--nocache"
)
$proc = Start-Process -FilePath $setup -ArgumentList $argList -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow -PassThru -Wait
Write-Output "Done at $(Get-Date). Exit code: $($proc.ExitCode)"
