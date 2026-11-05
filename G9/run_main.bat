@echo off
cd /d "%~dp0"
call E:\Dev\venv\Scripts\activate.bat
set GWEVA_DATA=E:\Dev\gweva-data
set LOGLEVEL=ERROR
set PYTHONWARNINGS=ignore
set TRANSFORMERS_VERBOSITY=error
set DATASETS_VERBOSITY=error
set HF_HUB_DISABLE_PROGRESS_BARS=1
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1

echo ========================================================
echo Starting G9 Main Experiment Runs (13 total)
echo ========================================================

:: 1. SST-2 (3 seeds, lr=1e-3)
echo.
echo [1/13] Running SST-2 seed 0...
python src\train.py --stage main --task sst2 --lr 1e-3 --seed 0
echo.
echo [2/13] Running SST-2 seed 1...
python src\train.py --stage main --task sst2 --lr 1e-3 --seed 1
echo.
echo [3/13] Running SST-2 seed 2...
python src\train.py --stage main --task sst2 --lr 1e-3 --seed 2

:: 2. MRPC (5 seeds, lr=2e-4)
echo.
echo [4/13] Running MRPC seed 0...
python src\train.py --stage main --task mrpc --lr 2e-4 --seed 0
echo.
echo [5/13] Running MRPC seed 1...
python src\train.py --stage main --task mrpc --lr 2e-4 --seed 1
echo.
echo [6/13] Running MRPC seed 2...
python src\train.py --stage main --task mrpc --lr 2e-4 --seed 2
echo.
echo [7/13] Running MRPC seed 3...
python src\train.py --stage main --task mrpc --lr 2e-4 --seed 3
echo.
echo [8/13] Running MRPC seed 4...
python src\train.py --stage main --task mrpc --lr 2e-4 --seed 4

:: 3. RTE (5 seeds, lr=2e-4)
echo.
echo [9/13] Running RTE seed 0...
python src\train.py --stage main --task rte --lr 2e-4 --seed 0
echo.
echo [10/13] Running RTE seed 1...
python src\train.py --stage main --task rte --lr 2e-4 --seed 1
echo.
echo [11/13] Running RTE seed 2...
python src\train.py --stage main --task rte --lr 2e-4 --seed 2
echo.
echo [12/13] Running RTE seed 3...
python src\train.py --stage main --task rte --lr 2e-4 --seed 3
echo.
echo [13/13] Running RTE seed 4...
python src\train.py --stage main --task rte --lr 2e-4 --seed 4

echo.
echo ========================================================
echo All G9 main runs finished!
echo ========================================================
pause
