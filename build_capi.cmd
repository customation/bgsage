@echo off
rem Build the C API shim with MSVC + Ninja (CPU-only), using the full
rem VS 2026 toolchain (vcvarsall.bat). cmake/ninja come from the
rem bgsage-worker venv.
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 || exit /b 1
set "PATH=%PATH%;C:\git\github\customation\bgsage-worker\.venv\Scripts"

cd /d C:\git\github\customation\bgsage
cmake -S cpp/capi -B build_capi_msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl || exit /b 1
ninja -C build_capi_msvc || exit /b 1
echo BUILD OK
