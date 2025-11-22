#!/usr/bin/env python3
"""
Script to check if all required packages from requirements.txt are installed.
"""
import sys

# Packages from requirements.txt
required_packages = [
    'imageio',
    'matplotlib',
    'numpy',
    'PyMCubes',
    'tqdm',
    'scipy',
    'plotly',
    'tensorboard',
]

# Also check for PyTorch and PyTorch3D since they're needed for the assignment
optional_but_important = [
    'torch',
    'pytorch3d',
]

print("=" * 60)
print("Checking required packages from requirements.txt")
print("=" * 60)

missing_packages = []
installed_packages = []

for package in required_packages:
    try:
        # Handle special cases for package names vs import names
        import_name = package
        if package == 'PyMCubes':
            import_name = 'mcubes'
        elif package == 'tensorboard':
            import_name = 'tensorboard'
        
        __import__(import_name)
        installed_packages.append(package)
        print(f"✓ {package} is installed")
    except ImportError:
        missing_packages.append(package)
        print(f"✗ {package} is MISSING")

print("\n" + "=" * 60)
print("Checking important packages (PyTorch, PyTorch3D)")
print("=" * 60)

for package in optional_but_important:
    try:
        __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"⚠ {package} is MISSING (may be needed for this assignment)")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if missing_packages:
    print(f"\n❌ Missing {len(missing_packages)} required package(s):")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print(f"\nTo install missing packages, you may need to:")
    print(f"   1. Activate your overlay and install: pip install {' '.join(missing_packages)}")
    print(f"   2. Or rebuild your container with these packages")
    sys.exit(1)
else:
    print("\n✅ All required packages are installed!")
    sys.exit(0)

