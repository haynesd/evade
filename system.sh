#!/bin/bash

echo "=============================="
echo "  SYSTEM INFORMATION SUMMARY  "
echo "=============================="

# Detect environment
KERNEL_NAME=$(uname -s)
IS_WSL=false
IS_GIT_BASH=false
IS_LINUX=false

case "$KERNEL_NAME" in
  *Microsoft*) IS_WSL=true ;;
  MINGW64_NT*|MSYS_NT*) IS_GIT_BASH=true ;;
  Linux) IS_LINUX=true ;;
esac

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
    grep -E '^NAME=|^VERSION=' /etc/os-release | cut -d= -f2 | tr -d '"'
elif $IS_GIT_BASH; then
    systeminfo | grep -E "OS Name|OS Version|BIOS Version"
fi

# Hardware & Firmware
echo
echo "-- Firmware and Hardware --"

if $IS_GIT_BASH; then
    echo "Using WMIC (Windows) tools..."

    echo
    echo "CPU:"
    wmic cpu get Name | grep -v "^$" | tail -n +2

    echo
    echo "Memory:"
    wmic memorychip get Capacity,Speed | grep -v "^$" | tail -n +2

    echo
    echo "BIOS:"
    wmic bios get Manufacturer,SMBIOSBIOSVersion,ReleaseDate | grep -v "^$" | tail -n +2

    echo
    echo "Motherboard:"
    wmic baseboard get Manufacturer,Product | grep -v "^$" | tail -n +2

    echo
    echo "Disks:"
    wmic diskdrive get Model,Size | grep -v "^$" | tail -n +2

    echo
    echo "GPU:"
    wmic path win32_VideoController get Name | grep -v "^$" | tail -n +2

elif $IS_WSL || $IS_LINUX; then
    echo
    echo "CPU:"
    lscpu | grep "Model name" || echo "lscpu not found"

    echo
    echo "Memory:"
    free -h || echo "free not found"

    echo
    echo "BIOS:"
    sudo dmidecode -t bios | grep -E "Vendor|Version|Release Date" || echo "dmidecode not found"

    echo
    echo "Motherboard:"
    sudo dmidecode -t baseboard | grep -E "Manufacturer|Product Name" || echo "dmidecode not found"

    echo
    echo "Disks:"
    lsblk || echo "lsblk not found"

    echo
    echo "GPU:"
    lspci | grep -i vga || echo "lspci not found"
fi

echo
echo "Done."
