#!/usr/bin/env bash
set -uo pipefail
# config
# colors
RED="\e[31m"
GREEN="\e[32m"
NC="\e[0m"

echo "========== unit test =========="
echo "========== clean =========="
make clean
echo "========== make ==========="
make test
echo "========== test ==========="
# normal cases
echo -n "normal cases..."
./sat.out
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

# edge cases
echo -n "polygon with only 2 vertices..."
echo -e "\
1\n\
2\n\
[(0.5443052896643373,0.7616839755359994),(0.9315831160478467,0.30600531904742123)]" > temp_test.txt
./sat.out temp_test.txt 2> /dev/null
if [ $? -ne 0 ]; then
    rm temp_test.txt
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi


# DEPRECATED. C code is unable to check this.
# echo -n "polygon with wrong vertices count..."
# echo -e "\
# 2\n\
# 4\n\
# [(0.17128805540516856,0.9272493837339879),(0.6967251294931065,0.1369653037373677),(0.7901952566001147,0.3064719963697743),(0.4750067345398874,0.9123178859928572)]\n\
# 4\n\
# [(0.9697434199841954,0.15800280893000296),(0.25335827371015274,0.9110765710404206),(0.38159895810972,0.06492942482722197)]" > temp_test.txt
# ./sat.out temp_test.txt 2> /dev/null
# if [ $? -ne 0 ]; then
#     rm temp_test.txt
#     echo -e "${GREEN}PASSED${NC}"
# else
#     echo -e "${RED}FAILED${NC}"
#     exit 1
# fi

echo ""
echo "========== module test =========="
echo "========== clean =========="
make clean
echo "========== make ==========="
make
echo "========== test ==========="
echo -n "test1 nonoverlap..."
./sat.out ./testcases/test1_nonoverlap.txt > ./testcases/tempoutput.txt
diff ./testcases/test1_nonoverlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "test2 overlap..."
./sat.out ./testcases/test2_overlap.txt > ./testcases/tempoutput.txt
diff ./testcases/test2_overlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "test3 nonoverlap..."
./sat.out ./testcases/test3_nonoverlap.txt > ./testcases/tempoutput.txt
diff ./testcases/test3_nonoverlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

rm ./testcases/tempoutput.txt
