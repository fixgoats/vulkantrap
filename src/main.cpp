
#include "hack.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#include "typedefs.hpp"
#include <complex>
#include <cxxopts.hpp>
#include <stdexcept>
#include <string>

#include "GPEsim.hpp"
#include "mathhelpers.hpp"
#include "rk4.h"
#include "vkhelpers.hpp"
#include <toml++/toml.hpp>

using std::bit_cast;

int main(int argc, char* argv[]) {
  constexpr u32 nElementsX = 256;
  constexpr u32 nElementsY = 256;
  constexpr u32 nElementsTotal = nElementsX * nElementsY;
  constexpr u32 maxSize = nElementsX * nElementsY * 16;
  VulkanApp myApp{maxSize};
  auto System = myApp.makeBuffer<cvec2>(nElementsTotal);
  auto S1 = myApp.makeBuffer<f32>(nElementsTotal);
  auto S2 = myApp.makeBuffer<f32>(nElementsTotal);
  auto S3 = myApp.makeBuffer<f32>(nElementsTotal);
  myApp.defaultInitBuffer<f32>(S1, nElementsTotal);
  myApp.defaultInitBuffer<f32>(S2, nElementsTotal);
  myApp.defaultInitBuffer<f32>(S3, nElementsTotal);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.01, 0.01);
  std::vector<cvec2> localSystem(nElementsTotal);
  for (auto& e : localSystem) {
    e = {{dis(gen), dis(gen)}, {dis(gen), dis(gen)}};
  }
  myApp.writeToBuffer(System, localSystem);
  struct RK4SpecConsts {
    u32 nX = nElementsX;
    u32 nY = nElementsY;
    u32 xSize = 8;
    u32 ySize = 8;
    float xStart = 0.0;
    float xEnd = 2.0;
    float eStart = 0.0;
    float eEnd = 1.0;
  } rk4SpecConsts;
  u32 blochSpecConsts[2] = {nElementsTotal, 64};
  struct DivSpecConsts {
    u32 size = nElementsTotal;
    u32 xgroup = 64;
    f32 divisor = 100000.0;
  } divSpecConsts;
  std::vector<u32> offsets{0, 0, 0, 0, 0, 0, 0, 0};
  Algorithm rk4 = myApp.makeAlgorithm("rk4sim.spv", {&System},
                                      bit_cast<u8*>(&rk4SpecConsts), offsets);
  Algorithm bloch =
      myApp.makeAlgorithm("s3.spv", {&System, &S1, &S2, &S3},
                          bit_cast<u8*>(&blochSpecConsts), {0, 0});
  Algorithm divbyscalarinplaces1 =
      myApp.makeAlgorithm("divinplacebyscalar.spv", {&S1},
                          bit_cast<u8*>(&divSpecConsts), {0, 0, 0});
  Algorithm divbyscalarinplaces2 =
      myApp.makeAlgorithm("divinplacebyscalar.spv", {&S2},
                          bit_cast<u8*>(&divSpecConsts), {0, 0, 0});
  Algorithm divbyscalarinplaces3 =
      myApp.makeAlgorithm("divinplacebyscalar.spv", {&S3},
                          bit_cast<u8*>(&divSpecConsts), {0, 0, 0});
  vk::CommandBuffer buffer = myApp.beginRecord();
  auto start = std::chrono::high_resolution_clock::now();
  // appendOpNoBarrier(buffer, rk4, 32, 32);
  for (uint32_t i = 0; i < 500; i++) {
    appendOp(buffer, rk4, 32, 32);
  }
  buffer.end();
  for (u32 i = 0; i < 200; i++) {
    myApp.execute(buffer);
  }
  myApp.device.freeCommandBuffers(myApp.commandPool, buffer);
  buffer = myApp.beginRecord();
  for (uint32_t i = 0; i < 500; i++) {
    appendOp(buffer, rk4, 32, 32);
    appendOp(buffer, bloch, 32, 32);
  }
  buffer.end();
  for (u32 i = 0; i < 200; i++) {
    myApp.execute(buffer);
  }
  myApp.device.freeCommandBuffers(myApp.commandPool, buffer);
  /*buffer = myApp.beginRecord();
  appendOp(buffer, divbyscalarinplaces1, 1024);
  appendOp(buffer, divbyscalarinplaces2, 1024);
  appendOp(buffer, divbyscalarinplaces3, 1024);
  buffer.end();
  myApp.execute(buffer);
  myApp.device.freeCommandBuffers(myApp.commandPool, buffer);*/
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << std::format("Simulation ran for {} ms\n", elapsed);
  myApp.writeFromBuffer(System, localSystem);
  std::vector<f32> s1(nElementsTotal);
  std::vector<f32> s2(nElementsTotal);
  std::vector<f32> s3(nElementsTotal);
  myApp.writeFromBuffer(S1, s1);
  myApp.writeFromBuffer(S2, s2);
  myApp.writeFromBuffer(S3, s3);
  std::ofstream of("system.csv");
  writeCsv(of, (c32*)localSystem.data(), 2 * nElementsX, nElementsY);
  of.open("S1.csv");
  writeCsv(of, (f32*)s1.data(), nElementsX, nElementsY);
  of.open("S2.csv");
  writeCsv(of, (f32*)s2.data(), nElementsX, nElementsY);
  of.open("S3.csv");
  writeCsv(of, (f32*)s3.data(), nElementsX, nElementsY);
}
