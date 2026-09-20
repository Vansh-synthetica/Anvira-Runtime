using System;
using System.Diagnostics;
using System.Text;

class LldLinkWrapper {
    static int Main(string[] args) {
        // lld-link wrapper that:
        //  1) injects the Windows SDK LIBPATH so kernel32.lib / ntdll.lib / ws2_32.lib etc. resolve
        //  2) substitutes msvcrt -> ucrt because the MSVC C runtime import library
        //     is not installed on this machine (only the Windows 10 SDK is)
        //  3) properly quote arguments so paths with spaces survive the round-trip
        string ucrt = @"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64";
        string um   = @"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64";
        string lld  = @"C:\Program Files\LLVM\bin\lld-link.exe";

        var sb = new StringBuilder();
        sb.Append("/LIBPATH:\"").Append(ucrt).Append("\" /LIBPATH:\"").Append(um).Append('"');
        foreach (string a in args) {
            // Handle both /defaultlib:msvcrt (lld-link style, no .lib) and
            // standalone msvcrt.lib (linker-wrapped style, with .lib)
            string arg = a.Replace("msvcrt.lib", "ucrt.lib")
                          .Replace("/defaultlib:msvcrt", "/defaultlib:ucrt");
            sb.Append(' ');
            if (arg.IndexOf(' ') >= 0) {
                sb.Append('"').Append(arg).Append('"');
            } else {
                sb.Append(arg);
            }
        }

        var psi = new ProcessStartInfo();
        psi.FileName  = lld;
        psi.Arguments = sb.ToString();
        psi.UseShellExecute = false;
        var p = Process.Start(psi);
        p.WaitForExit();
        return p.ExitCode;
    }
}
