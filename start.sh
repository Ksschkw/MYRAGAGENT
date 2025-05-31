#!/bin/bash
set -e  # Exit on error
echo "Starting RAG Agent..."
python AgentKosiV2.py --server || { echo "Failed to start server"; exit 1; }