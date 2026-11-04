#!/usr/bin/env python3
"""
Entry point untuk menjalankan Benchmark EAR dari project root.
Mengarahkan eksekusi ke scripts/benchmark_ear.py
"""

import sys
import os

scripts_path = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, scripts_path)

from benchmark_ear import main

if __name__ == '__main__':
    main()
