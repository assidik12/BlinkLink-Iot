#!/usr/bin/env python3
"""
Entry point untuk menjalankan BlinkLink-IoT dari project root.
Ini adalah wrapper yang menjalankan src/main.py dengan path yang tepat.
"""

import sys
import os

# Tambahkan src directory ke Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import dan jalankan main
from main import main

if __name__ == '__main__':
    main()
