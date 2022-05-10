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
echo "========== start ==========="
./sat.out
if [ $? -ne 0 ]; then
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo ""
echo "========== module test =========="
echo "========== clean =========="
make clean
echo "========== make ==========="
make
echo "========== start ==========="

# DEPRECATED: C code is unable to check this.
# echo -n "polygon with wrong vertices count..."
# echo -e "\
# 2\n\
# 4\n\
# [(0.17128805540516856,0.9272493837339879),(0.6967251294931065,0.1369653037373677),(0.7901952566001147,0.3064719963697743),(0.4750067345398874,0.9123178859928572)]\n\
# 4\n\
# [(0.9697434199841954,0.15800280893000296),(0.25335827371015274,0.9110765710404206),(0.38159895810972,0.06492942482722197)]" > temp_test.txt
# ./sat.out -i temp_test.txt 2> /dev/null
# if [ $? -ne 0 ]; then
#     rm temp_test.txt
#     echo -e "${GREEN}PASSED${NC}"
# else
#     echo -e "${RED}FAILED${NC}"
#     exit 1
# fi

echo -n "polygon with only 2 vertices..."
echo -e "\
1\n\
2\n\
[(0.5443052896643373,0.7616839755359994),(0.9315831160478467,0.30600531904742123)]" > temp_test.txt
./sat.out -i temp_test.txt 2> /dev/null
# should fail due to invalid polygon.
if [ $? -ne 0 ]; then
    rm temp_test.txt
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "non existing file..."
<<<<<<< HEAD
./sat.out -i ./wtf 2> /dev/null
=======
./sat.out -i ./wtf > /dev/null
>>>>>>> 2f8b7fd7f622def08538960ff0d5ef4ba66320b4
# should fail due to input file not existing.
if [ $? -ne 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "test1 nonoverlap..."
<<<<<<< HEAD
./sat.out -i ./testcases/test1_nonoverlap.txt -o ./testcases/tempoutput.txt > /dev/null
diff ./testcases/test1_nonoverlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "test1 nonoverlap(GPU)..."
./sat.out -i ./testcases/test1_nonoverlap.txt -o ./testcases/tempoutput.txt -g > /dev/null
=======
./sat.out -i ./testcases/test1_nonoverlap.txt > ./testcases/tempoutput.txt
>>>>>>> 2f8b7fd7f622def08538960ff0d5ef4ba66320b4
diff ./testcases/test1_nonoverlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "test2 overlap..."
<<<<<<< HEAD
./sat.out -i ./testcases/test2_overlap.txt -o ./testcases/tempoutput.txt > /dev/null
diff ./testcases/test2_overlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "test2 overlap(GPU)..."
./sat.out -i ./testcases/test2_overlap.txt -o ./testcases/tempoutput.txt -g > /dev/null
=======
./sat.out -i ./testcases/test2_overlap.txt > ./testcases/tempoutput.txt
>>>>>>> 2f8b7fd7f622def08538960ff0d5ef4ba66320b4
diff ./testcases/test2_overlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "test3 nonoverlap..."
<<<<<<< HEAD
./sat.out -i ./testcases/test3_nonoverlap.txt -o ./testcases/tempoutput.txt > /dev/null
diff ./testcases/test3_nonoverlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "test3 nonoverlap(GPU)..."
./sat.out -i ./testcases/test3_nonoverlap.txt -o ./testcases/tempoutput.txt -g > /dev/null
=======
./sat.out -i ./testcases/test3_nonoverlap.txt > ./testcases/tempoutput.txt
>>>>>>> 2f8b7fd7f622def08538960ff0d5ef4ba66320b4
diff ./testcases/test3_nonoverlap_res.txt ./testcases/tempoutput.txt > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

rm ./testcases/tempoutput.txt
