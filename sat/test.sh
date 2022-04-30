#!/usr/bin/env bash
set -uo pipefail

echo "========== clean =========="
make clean
echo "========== make ==========="
make test
echo "========== test ==========="
# normal cases
./sat.out
if [ $? -eq 0 ]; then
    echo "PASSED"
else
    echo "FAILED"
fi

# edge cases
echo "\
1\n\
2\n\
[(0.5443052896643373,0.7616839755359994),(0.9315831160478467,0.30600531904742123)]" > temp_test.txt
./sat.out temp_test.txt 2> /dev/null
if [ $? -ne 0 ]; then
    echo "PASSED"
else
    echo "FAILED"
fi
