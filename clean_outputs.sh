#!/bin/bash

echo "Cleaning up simulation outputs..."

if [ -d "analysis_outputs" ]; then
    echo "  Removing analysis_outputs/ directory..."
    rm -rf analysis_outputs/
    echo "  analysis_outputs/ removed"
else
    echo "  analysis_outputs/ directory not found"
fi

if [ -f "cbm_py_output.npz" ]; then
    echo "  Removing cbm_py_output.npz..."
    rm -f cbm_py_output.npz
    echo "  cbm_py_output.npz removed"
else
    echo "  cbm_py_output.npz not found"
fi

if [ -f "*.png" ]; then
    echo "  Removing PNG files..."
    rm -f *.png
    echo "  PNG files removed"
fi

if [ -f "*.npz" ]; then
    echo "  Removing other NPZ files..."
    rm -f *.npz
    echo "  NPZ files removed"
fi

echo "Cleanup complete. Ready for a fresh simulation run."
