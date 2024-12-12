## Build instructions
Required dependencies: Vulkan, opencv (will maybe ditch this eventually for an
actual plotting library), glslang, VulkanMemoryAllocator, flatc + flatbuffers libs,

```
cmake -B build
./compileEverything.sh
```
You may need to make `compileEverything.sh` executable: `chmod +x compileEverything.sh`.

## To run
Subject to change. Run the simulation and show the zeeman splitting.
```
./VulkanCompute -c
```
Debug flag that checks whatever I'm trying to figure out at the time:
```
./VulkanCompute -d
```

## To Do
Make actual plots with colorbars, export/plot more quantities. Finish helper
functions to save and load tensors.
