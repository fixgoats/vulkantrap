#!/bin/bash
#flatc -o include/ --cpp floatmat.fbs
#flatc -o include/ --cpp doublemat.fbs
#flatc -o include/ --cpp floatarr3.fbs
#flatc -o include/ --cpp doublearr3.fbs
cmake --build build --parallel $(nproc)
glslangValidator -V shaders/rk4sim.comp -o build/rk4sim.spv
