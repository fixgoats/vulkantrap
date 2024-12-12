#!/bin/bash
cmake --build Debug --parallel
glslangValidator -V shaders/rk4sim.comp -o Debug/rk4sim.spv
glslangValidator -V shaders/ediff.comp -o Debug/ediff.spv
