#!/usr/bin/env python3
"""
Quick start script for E-commerce Product Recommender
This script will start the Flask application with the web interface
"""

import os
import sys
import webbrowser
import time
import subprocess
from threading import Timer

def open_browser():
    """Open the web browser after a short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def main():
    print("=" * 60)
    print("E-commerce Product Recommender")
    print("=" * 60)
    print()
    print("Starting the application...")
    print("The web interface will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Open browser after 2 seconds
    Timer(2.0, open_browser).start()
    
    # Start the Flask application
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run([sys.executable, 'backend/app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped. Thank you for using the E-commerce Product Recommender!")
    except subprocess.CalledProcessError as e:
        print(f"Error starting the server: {e}")
        print("Make sure you have activated the virtual environment and installed dependencies.")

if __name__ == '__main__':
    main()
