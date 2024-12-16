#include "hack.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <random>

#include "typedefs.hpp"
#include "vkFFT.h"
#include <complex>
#include <cxxopts.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#include "GPEsim.hpp"
#include "mathhelpers.hpp"
#include "vkhelpers.hpp"
#include <toml++/toml.hpp>
#include <type_traits>

#define PRINTVAR(x) std::cout << x << '\n'

using namespace std::complex_literals;

std::string tstamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t time = std::chrono::system_clock::to_time_t(now);
  tm local_tm = *localtime(&time);
  return std::format("{}-{}-{}/{}-{}", local_tm.tm_year - 100,
                     local_tm.tm_mon + 1, local_tm.tm_mday, local_tm.tm_hour,
                     local_tm.tm_min);
}

SimConstants coupledConfig(const toml::table& tbl) {
  return {tbl["nElementsX"].value_or(256u), tbl["nElementsY"].value_or(256u),
          tbl["times"].value_or(256u),      tbl["xGroupSize"].value_or(8u),
          tbl["yGroupSize"].value_or(8u),   tbl["xstart"].value_or(0.0f),
          tbl["xstart"].value_or(0.0f),     tbl["estart"].value_or(0.0f),
          tbl["eend"].value_or(0.0f)};
}

SimConstants simpleConfig(const toml::table& tbl) {
  return {tbl["nElementsX"].value_or(256u), 1,  tbl["times"].value_or(100000u),
          tbl["xGroupSize"].value_or(8u),   1,  tbl["xstart"].value_or(0.0f),
          tbl["xend"].value_or(0.0f),       0., 0.};
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("GPEsim",
                           "Vulkan simulation of Gross-Pitaevskii equation");
  options.add_options()("c,config", "TOML config file",
                        cxxopts::value<std::string>())(
      "s,scan", "TOML config file", cxxopts::value<std::string>())(
      "m,model", "Use simpler model", cxxopts::value<std::string>());
  auto result = options.parse(argc, argv);
  if (result.count("c")) {
    toml::table tbl{};
    auto infile = result["c"].as<std::string>();
    tbl = toml::parse_file(infile);
    auto sc = coupledConfig(*tbl["constants"].as_table());

    VulkanApp app{sc};
    std::cout << "Initialized GPE fine\n";
    app.initBuffers();
    std::cout << "Uploaded data\n";
    auto start = std::chrono::high_resolution_clock::now();
    app.runSim(0);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << std::format("Simulation ran for: {} ms\n", elapsed.count());
    auto system = app.outputBuffer<cvec2>(0);
    std::ofstream values;
    auto dir = std::format("data/{}", tstamp());
    std::filesystem::create_directories(dir);
    std::filesystem::copy(infile, dir);
    values.open(std::format("{}/psip.csv", dir));
    for (uint32_t j = 0; j < app.params.nElementsY; ++j) {
      for (uint32_t i = 0; i < app.params.nElementsX; ++i) {
        values << std::format(
            " {}", numfmt(system[j * app.params.nElementsX + i].psip));
      }
      values << '\n';
    }
    values.close();
    values.open(std::format("{}/psim.csv", dir));
    for (uint32_t j = 0; j < app.params.nElementsY; ++j) {
      for (uint32_t i = 0; i < app.params.nElementsX; ++i) {
        values << std::format(
            " {}", numfmt(system[j * app.params.nElementsX + i].psim));
      }
      values << '\n';
    }
    values.close();
    return 0;

  } else if (result.count("m")) {
    toml::table tbl{};
    auto infile = result["m"].as<std::string>();
    tbl = toml::parse_file(infile);
    auto sc = simpleConfig(*tbl["simpler"].as_table());

    VulkanApp app{sc};
    std::cout << "Initialized GPE fine\n";
    app.initBuffers();
    std::cout << "Uploaded data\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto dir = std::format("data/{}", tstamp());
    std::filesystem::create_directories(dir);
    std::filesystem::copy(infile, dir);
    std::ofstream values;
    std::ofstream othervalues;
    values.open(std::format("{}/psi.csv", dir));
    for (uint32_t i = 0; i < 1000; i++) {
      app.runSim(1);
      auto system = app.outputBuffer<cvec2>(0);
      for (uint32_t i = 0; i < app.params.nElementsX; ++i) {
        values << std::format(" {}", numfmt(system[i].psip));
        values << std::format(" {}", numfmt(system[i].psim));
      }
      values << '\n';
    }
    values.close();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << std::format("Simulation ran for: {} ms\n", elapsed.count());
    return 0;

  } else {
    throw std::runtime_error("gib c\n");
  }
}
