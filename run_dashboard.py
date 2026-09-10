"""
run_dashboard.py

1-Click launcher for the Godrej Field Intelligence Dashboard & AI Backend.
Starts the FastAPI server with all endpoints, serves the 6 Stitch pages,
and opens http://localhost:8000 in your browser automatically.
"""

import os
import sys
import time
import webbrowser
import uvicorn

def main():
    print("=" * 60)
    print("  FIELD INTELLIGENCE // WAREHOUSE VIDEO INTELLIGENCE")
    print("  Godrej Hackathon - Person D Dashboard & Copilot Backend")
    print("=" * 60)
    print("\n[+] Starting FastAPI server on http://localhost:8000...")
    print("[+] Dashboard views available:")
    print("    - Overview:     http://localhost:8000/overview.html")
    print("    - Intelligence: http://localhost:8000/intelligence.html")
    print("    - Incidents:    http://localhost:8000/incidents.html")
    print("    - Behaviour:    http://localhost:8000/behaviour.html")
    print("    - Prevention:   http://localhost:8000/prevention.html")
    print("    - AI Assistant: http://localhost:8000/assistance.html")
    print("\n[+] Opening dashboard in browser in 1.5s...")

    # Open browser after a brief delay so server has time to bind
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000/overview.html")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
