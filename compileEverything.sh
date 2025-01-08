#!/bin/bash
#flatc -o include/ --cpp floatmat.fbs
#flatc -o include/ --cpp doublemat.fbs
#flatc -o include/ --cpp floatarr3.fbs
#flatc -o include/ --cpp doublearr3.fbs
cmake --build build --parallel $(nproc)
glslangValidator -V shaders/rk4sim.comp -o build/rk4sim.spv
glslangValidator -V shaders/simplermodel.comp -o build/simplermodel.spv
glslangValidator -V shaders/s3.comp -o build/s3.spv
glslangValidator -V shaders/divinplacebyscalar.comp -o build/divinplacebyscalar.spv
