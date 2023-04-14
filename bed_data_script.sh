#!/bin/bash

if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
    source env/bin/activate
    pip3 install -r requirements.txt
else
    echo "Virtual environment already exists."
    source env/bin/activate
fi

python3 load_bed_data.py