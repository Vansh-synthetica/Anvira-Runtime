@echo off
rem Guided tour of the Anvira CLI. Run it from the runtime folder:   examples\cli_demo.cmd   (or copy it next to anvira.cmd)
rem Every step is a real command. Nothing is downloaded and no model is installed; it only reads and, at the end, uses a model YOU pass in.
setlocal
set "A=%~dp0..\anvira.cmd"
if not exist "%A%" set "A=%~dp0anvira.cmd"
set "MODELS=%~1"

echo.
echo === 1. Where is the runtime, and is it healthy? ==========================
call "%A%" runtime locate
call "%A%" doctor
echo.
echo === 2. Start it, see it ====================================================
call "%A%" runtime start
call "%A%" status
call "%A%" hardware
echo.
echo === 3. Models: put them anywhere, the runtime remembers ====================
if not "%MODELS%"=="" call "%A%" model dirs add "%MODELS%"
call "%A%" model list --installed
call "%A%" model recommend
echo.
echo === 4. Apps and permissions ================================================
call "%A%" app list
call "%A%" capability list
echo.
echo === 5. Shared data between apps (private by default) =======================
call "%A%" context add "Demo note" --text "The Calvin cycle fixes carbon dioxide using RuBisCO." --owner anvira-notes
call "%A%" context list
call "%A%" context search "which enzyme fixes carbon dioxide"
call "%A%" context audit
echo.
echo === 6. The dashboard pages (one frame each) ===============================
call "%A%" open --once
call "%A%" open health --once
call "%A%" open models --once
call "%A%" open apps --once
echo.
echo === 7. Use a model (pass its id as the 2nd argument) ======================
if not "%~2"=="" (
  call "%A%" model use %~2
  call "%A%" chat "In one sentence, what is a hash map?" --no-stream
  call "%A%" orcha run "List three uses of a queue" --no-wait
  call "%A%" orcha jobs
)
echo.
echo === 8. Stop (apps normally do this by themselves when they close) =========
call "%A%" runtime stop
echo Done. Try the live dashboard:   anvira open
