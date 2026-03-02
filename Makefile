all: utils gaas

utils: utils.cpp

	g++ utils.cpp -o utils

gaas: *.cu *.h

	nvcc -O3 -arch=sm_86 -G -lcurand -maxrregcount=64 --expt-extended-lambda *.cu -o gaas
