@echo off
cd /d "C:\Users\dougl\DougDirectorComfyui\DougDirectorComfyUI"
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 7860
