#!/usr/bin/env python3
"""
Installation script to resolve dependency conflicts.
Installs production dependencies first, then development dependencies.
"""

import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None

def main():
    """Install dependencies in the correct order to avoid conflicts."""
    print("🚀 Installing Diabetes Prediction MLOps dependencies...")
    
    # Step 1: Upgrade pip
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install production dependencies with exact versions
    print("\n📦 Installing production dependencies...")
    result = run_command("pip install -r requirements.txt --no-cache-dir", "Installing production requirements")
    
    if result is None:
        print("❌ Failed to install production dependencies")
        sys.exit(1)
    
    # Step 3: Check for conflicts after production install
    print("\n🔍 Checking for dependency conflicts...")
    conflict_check = subprocess.run("pip check", shell=True, capture_output=True, text=True)
    
    if conflict_check.returncode != 0:
        print("⚠️  Dependency conflicts detected:")
        print(conflict_check.stdout)
        print(conflict_check.stderr)
        print("\n🔧 Attempting to resolve conflicts...")
        
        # Try to install specific compatible versions
        run_command("pip install pydantic==2.10.1 --force-reinstall --no-deps", "Force installing Pydantic 2.10.1")
        run_command("pip install prefect==3.4.11 --no-deps", "Installing Prefect without deps")
    else:
        print("✅ No dependency conflicts found")
    
    # Step 4: Install development dependencies (optional)
    if len(sys.argv) > 1 and sys.argv[1] == "--dev":
        print("\n📦 Installing development dependencies...")
        dev_result = run_command("pip install -r requirements-dev.txt", "Installing development requirements")
        
        if dev_result is None:
            print("⚠️  Development dependencies failed, but production deps are OK")
    
    # Step 5: Final conflict check
    print("\n🔍 Final dependency check...")
    final_check = subprocess.run("pip check", shell=True, capture_output=True, text=True)
    
    if final_check.returncode == 0:
        print("✅ All dependencies installed successfully!")
        
        # Show installed versions
        print("\n📋 Key package versions:")
        key_packages = ["prefect", "pydantic", "fastapi", "mlflow", "evidently"]
        for package in key_packages:
            version_result = subprocess.run(f"pip show {package} | grep Version", 
                                          shell=True, capture_output=True, text=True)
            if version_result.returncode == 0:
                print(f"  {package}: {version_result.stdout.strip()}")
    else:
        print("❌ Dependency conflicts remain:")
        print(final_check.stdout)
        print(final_check.stderr)
        print("\n💡 Try running: pip install --force-reinstall pydantic==2.10.1")
        sys.exit(1)

if __name__ == "__main__":
    main()