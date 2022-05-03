#!/usr/bin/env bash
set -exuo pipefail

cd polygon_io
pipenv run python3 generator.py --count 500 --point 30000 --vmin 10 --vmax 30 --output polygon_input.txt
../sat/sat.out ./polygon_input.txt > ./polygon_output.txt
pipenv run python3 viewer.py --input polygon_output.txt --output draw.png
cd -
