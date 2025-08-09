#!/usr/bin/env python3
"""
Startup script for Forensic Backend Server
Author: Kelly-Ann Harris
Date: 2024
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent))
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 