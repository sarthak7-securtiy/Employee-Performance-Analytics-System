#!/usr/bin/env python3
"""
Initialize the database for the Automated Training Evaluation & Performance Analytics System.
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import init_db

def main():
    """Initialize the database"""
    print("Initializing database...")
    
    try:
        session = init_db()
        session.close()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing database: {str(e)}")

if __name__ == "__main__":
    main()