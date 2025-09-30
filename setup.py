#!/usr/bin/env python3
"""
Setup script for Tesla ML Trading Strategy
Quick installation and environment setup
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not compatible. Need Python 3.8+")
        return False

def setup_environment():
    """Setup the project environment"""
    print("🚀 Tesla ML Trading Strategy Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not os.path.exists('venv'):
        if not run_command('python -m venv venv', 'Creating virtual environment'):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Activate and install requirements
    if os.name == 'nt':  # Windows
        activate_cmd = 'venv\\Scripts\\activate'
        pip_cmd = 'venv\\Scripts\\pip'
    else:  # Unix/Linux/Mac
        activate_cmd = 'source venv/bin/activate'
        pip_cmd = 'venv/bin/pip'
    
    # Install requirements
    if not run_command(f'{pip_cmd} install -r requirements.txt', 'Installing dependencies'):
        return False
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    print("✅ Created data directories")
    
    # Create .env template if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# Tesla ML Trading Strategy Environment Variables\n")
            f.write("# Get your free API key at: https://newsapi.org/\n")
            f.write("NEWS_API_KEY=your_newsapi_key_here\n")
        print("✅ Created .env template file")
        print("📝 IMPORTANT: Edit .env file and add your NewsAPI key!")
    else:
        print("✅ .env file already exists")
    
    return True

def run_quick_test():
    """Run a quick test to verify setup"""
    print("\n🧪 Running quick setup verification...")
    
    # Test imports
    test_script = """
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import transformers
    import sklearn
    import yfinance as yf
    print("✅ All core dependencies imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
    
# Test basic functionality
print("✅ Setup verification passed!")
"""
    
    with open('test_setup.py', 'w') as f:
        f.write(test_script)
    
    if os.name == 'nt':  # Windows
        python_cmd = 'venv\\Scripts\\python'
    else:  # Unix/Linux/Mac
        python_cmd = 'venv/bin/python'
    
    success = run_command(f'{python_cmd} test_setup.py', 'Testing setup')
    
    # Clean up test file
    if os.path.exists('test_setup.py'):
        os.remove('test_setup.py')
    
    return success

def main():
    """Main setup function"""
    if setup_environment():
        if run_quick_test():
            print("\n🎉 Setup completed successfully!")
            print("\n📋 Next steps:")
            print("   1. Edit .env file and add your NewsAPI key")
            print("   2. Run: python tesla_ml_pipeline/modules/collect_news.py")
            print("   3. Follow the README.md for full pipeline usage")
            print("\n🚀 Happy trading!")
        else:
            print("\n⚠️  Setup completed but verification failed")
            print("   Check the error messages above and try manual installation")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()