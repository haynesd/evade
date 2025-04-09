#!/bin/bash

echo "=============================="
echo "  SYSTEM INFORMATION SUMMARY  "
echo "=============================="

# Detect environment
KERNEL_NAME=$(uname -s)
IS_WSL=false
IS_GIT_BASH=false
IS_LINUX=false

# Use /proc/version for WSL check
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
elif [[ "$KERNEL_NAME" == MINGW64_NT* || "$KERNEL_NAME" == MSYS_NT* ]]; then
    IS_GIT_BASH=true
elif [[ "$KERNEL_NAME" == "Linux" ]]; then
    IS_LINUX=true
fi

# Environment info
echo "-- Environment Detection --"
if $IS_WSL; then
    echo "Running on: WSL (Windows Subsystem for Linux)"
elif $IS_GIT_BASH; then
    echo "Running on: Git Bash (Windows)"
elif $IS_LINUX; then
    echo "Running on: Native Linux"
else
    echo "Environment: Unknown"
fi

# OS Info
echo
echo "-- Operating System --"
uname -a
if [[ -f /etc/os-release ]]; then
    grep -E '^PRETTY_NAME=|^VERSION=' /etc/os-release | cut -d= -f2 | tr -d '"'
elif $IS_GIT_BASH; then
    systeminfo | grep -E "OS Name|OS Version|BIOS Version"
fi

# Firmware & Hardware Info
echo
echo "-- Firmware and Hardware --"

if $IS_GIT_BASH; then
    echo "Using WMIC (Windows) tools..."

    echo -e "\nCPU:"
    wmic cpu get Name | grep -v "^$" | tail -n +2

    echo -e "\nMemory:"
    wmic memorychip get Capacity,Speed | grep -v "^$" | tail -n +2

    echo -e "\nBIOS:"
    wmic bios get Manufacturer,SMBIOSBIOSVersion,ReleaseDate | grep -v "^$" | tail -n +2

    echo -e "\nMotherboard:"
    wmic baseboard get Manufacturer,Product | grep -v "^$" | tail -n +2

    echo -e "\nDisks:"
    wmic diskdrive get Model,Size | grep -v "^$" | tail -n +2

    echo -e "\nGPU:"
    wmic path win32_VideoController get Name | grep -v "^$" | tail -n +2

else
    echo -e "\nCPU:"
    if command -v lscpu &>/dev/null; then
        lscpu | grep "Model name" || grep "Architecture" || echo "CPU info not found"
    else
        echo "lscpu not found"
    fi

    echo -e "\nMemory:"
    free -h || echo "free not found"

    echo -e "\nBIOS:"
    if command -v dmidecode &>/dev/null; then
        sudo dmidecode -t bios | grep -E "Vendor|Version|Release Date"
    else
        echo "dmidecode not available (likely restricted in WSL)"
    fi

    echo -e "\nMotherboard:"
    if command -v dmidecode &>/dev/null; then
        sudo dmidecode -t baseboard | grep -E "Manufacturer|Product Name"
    else
        echo "dmidecode not available (likely restricted in WSL)"
    fi

    echo -e "\nGPU:"
    if command -v lspci &>/dev/null; then
        lspci | grep -i vga || echo "No VGA-compatible GPU found"
    else
        echo "lspci not available (not supported in many WSL/ARM environments)"
    fi
fi

echo
echo "Done."
