#!/bin/bash

echo "=========================================="
echo "Starting Cerebellar Microcircuit Simulation"
echo "=========================================="

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

echo "Step 1: Running simulation..."
echo "Command: python3 main.py"
python3 main.py

if [ $? -eq 0 ]; then
    echo "Simulation completed successfully"
else
    echo "Simulation failed with exit code $?"
    exit 1
fi

echo ""
echo "Step 2: Running analysis..."
echo "Command: python3 analysis.py"
python3 analysis.py

if [ $? -eq 0 ]; then
    echo "Analysis completed successfully"
    echo ""
    echo "=========================================="
    echo "Simulation and Analysis Complete!"
    echo "=========================================="
    echo "Results saved to:"
    echo "  - Simulation data: cbm_py_output.npz"
    echo "  - Analysis plots: analysis_outputs/"
    echo ""
    echo "Generated plots:"
    echo "  - pf_raster.png"
    echo "  - io_raster.png"
    echo "  - pkj_raster.png"
    echo "  - dcn_raster.png"
    echo "  - pf_pkj_weight.png"
    echo "=========================================="
else
    echo "Analysis failed with exit code $?"
    exit 1
fi
