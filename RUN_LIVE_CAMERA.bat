@echo off
title Godrej AI - Live Warehouse Intelligence (Person A + B)
echo ======================================================================
echo  GODREJ WAREHOUSE AI - LIVE CAMERA
echo  Person A : Perception + Skeleton + Box Tracking
echo  Person B : Risk Engine + Near-Miss Alerts (Method 1)
echo ======================================================================
echo  Controls inside the live window:
echo    Q or ESC  ^>  Stop and save session
echo    S         ^>  Take a snapshot
echo ======================================================================
python run_live_end_to_end.py --camera 0
pause
