#include "hack.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
          tbl["xend"].value_or(0.0f),       tbl["estart"].value_or(0.0f),
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
      "m,model", "Use simpler model", cxxopts::value<std::string>())(
      "d,debug", "Test S3",
      cxxopts::value<std::string>())("s,sample",
                                     "Solve fewer systems on CPU, random "
                                     "coefficients in possibly chaotic range",
                                     cxxopts::value<std::string>())(
      "p,pew", "Solve with fixed coefficients but random initial conditions",
      cxxopts::value<std::string>())(
      "a,aaa", "Solve with fixed initial conditions and x, but scan over e",
      cxxopts::value<std::string>());
  auto result = options.parse(argc, argv);
  if (result.count("c")) {
    toml::table tbl{};
    auto infile = result["c"].as<std::string>();
    std::cout << "Got here\n";
    tbl = toml::parse_file(infile);
    std::cout << "Got here\n";
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
    std::filesystem::copy(infile, dir,
                          std::filesystem::copy_options::overwrite_existing);
    values.open(std::format("{}/psip.csv", dir));
    for (uint32_t j = 0; j < app.params.nElementsY; ++j) {
      for (uint32_t i = 0; i < app.params.nElementsX; ++i) {
        values << std::format(" {}",
                              numfmt(system[j * app.params.nElementsX + i].x));
      }
      values << '\n';
    }
    values.close();
    values.open(std::format("{}/psim.csv", dir));
    for (uint32_t j = 0; j < app.params.nElementsY; ++j) {
      for (uint32_t i = 0; i < app.params.nElementsX; ++i) {
        values << std::format(" {}",
                              numfmt(system[j * app.params.nElementsX + i].y));
      }
      values << '\n';
    }
    values.close();
    return 0;

  } else if (result.count("m")) {
    toml::table tbl{};
    auto infile = result["m"].as<std::string>();
    tbl = toml::parse_file(infile);
    auto sc = coupledConfig(*tbl["constants"].as_table());

    VulkanApp app{sc};
    std::cout << "Initialized GPE fine\n";
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> bleh(sc.elementsTotal(), 0);
    std::cout << "allocated " << 8 * sc.elementsTotal() << " bytes\n";
    std::vector<float> s3(sc.elementsTotal(), 0);
    std::cout << "allocated " << 4 * sc.elementsTotal() << " bytes\n";
    for (uint32_t i = 0; i < 10; i++) {
      app.initBuffers();
      app.runSim(0);
      app.s3();
      s3 = app.outputBuffer<float>(1);
      std::transform(s3.cbegin(), s3.cend(), bleh.begin(), bleh.begin(),
                     [](float a, double b) { return (double)a + b; });
    }
    std::transform(bleh.begin(), bleh.end(), bleh.begin(),
                   [](double x) { return x / 10.; });
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << std::format("Simulation ran for: {} ms\n", elapsed.count());
    auto dir = std::format("data/{}", tstamp());
    std::filesystem::create_directories(dir);
    std::filesystem::copy(infile, dir,
                          std::filesystem::copy_options::overwrite_existing);
    std::ofstream values;
    std::ofstream othervalues;
    values.open(std::format("{}/S3.csv", dir));
    for (uint32_t j = 0; j < sc.nElementsY; j++) {
      for (uint32_t i = 0; i < sc.nElementsX; i++) {
        values << std::format(" {}", numfmt(s3[j * sc.nElementsX + i]));
      }
      values << '\n';
    }
    values.close();
    return 0;

  } else if (result.count("d")) {
    toml::table tbl{};
    auto infile = result["d"].as<std::string>();
    tbl = toml::parse_file(infile);
    auto sc = coupledConfig(*tbl["constants"].as_table());

    VulkanApp app{sc};
    std::cout << "Initialized GPE fine\n";
    auto start = std::chrono::high_resolution_clock::now();
    app.tests3();
    auto s3 = app.outputBuffer<float>(1);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << std::format("Simulation ran for: {} ms\n", elapsed.count());
    auto dir = std::format("data/{}", tstamp());
    std::filesystem::create_directories(dir);
    std::filesystem::copy(infile, dir,
                          std::filesystem::copy_options::overwrite_existing);
    std::ofstream values;
    std::ofstream othervalues;
    values.open(std::format("{}/S3.csv", dir));
    for (uint32_t j = 0; j < sc.nElementsY; j++) {
      for (uint32_t i = 0; i < sc.nElementsX; i++) {
        values << std::format(" {}", numfmt(s3[j * sc.nElementsX + i]));
      }
      values << '\n';
    }
    values.close();
    return 0;

  } else if (result.count("s")) {
    toml::table tbl{};
    auto infile = result["s"].as<std::string>();
    tbl = toml::parse_file(infile);
    auto sc = coupledConfig(*tbl["constants"].as_table());

    auto dir = std::format("data/{}", tstamp());
    std::filesystem::create_directories(dir);
    std::filesystem::copy(infile, dir,
                          std::filesystem::copy_options::overwrite_existing);
    std::ofstream values;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.01, 0.01);
    {
      std::vector<float> xs(40);
      std::vector<float> es(40);
      std::vector<cvec2> psis(40);
      std::uniform_real_distribution<float> xdis(0.0, 1.0);
      for (auto& x : psis) {
        x = cvec2{{dis(gen), dis(gen)}, {dis(gen), dis(gen)}};
      }
      for (uint32_t i = 0; i < 40; i++) {
        float tmpx = xdis(gen);
        float tmpe = 0.2 * (tmpx + xdis(gen) * tmpx);
        xs[i] = tmpx;
        es[i] = tmpe;
      }
      values.open(std::format("{}/fullrandom.csv", dir));
      for (uint32_t i = 0; i < sc.times / 100; i++) {
        for (size_t j = 0; j < 40; j++) {
          for (uint32_t k = 0; k < 100; k++) {
            psis[j] = rk4(psis[j], 0.001f, xs[j], es[j]);
          }
          values << std::format(" {} {}", numfmt(psis[j].x), numfmt(psis[j].y));
        }
        values << '\n';
      }
      values.close();
    }
    {
      const float x = 0.3;
      const float e = 0.2;
      std::vector<cvec2> psis(40);
      for (auto& r : psis) {
        r = cvec2{{dis(gen), dis(gen)}, {dis(gen), dis(gen)}};
      }
      values.open(std::format("{}/randomic.csv", dir));
      for (uint32_t i = 0; i < sc.times / 100; i++) {
        for (size_t j = 0; j < 40; j++) {
          for (uint32_t k = 0; k < 100; k++) {
            psis[j] = rk4(psis[j], 0.001f, x, e);
          }
          values << std::format(" {} {}", numfmt(psis[j].x), numfmt(psis[j].y));
        }
        values << '\n';
      }
      values.close();
    }
    {
      const float x = 0.4;
      std::vector<float> es(40);
      std::vector<cvec2> psis(40, {{dis(gen), dis(gen)}, {dis(gen), dis(gen)}});
      for (uint32_t i = 0; i < 40; i++) {
        es[i] = 0.1 + (float)i * 0.01;
      }
      values.open(std::format("{}/escan.csv", dir));
      for (uint32_t i = 0; i < sc.times / 100; i++) {
        for (size_t j = 0; j < 40; j++) {
          for (uint32_t k = 0; k < 100; k++) {
            psis[j] = rk4(psis[j], 0.001f, x, es[j]);
          }
          values << std::format(" {} {}", numfmt(psis[j].x), numfmt(psis[j].y));
        }
        values << '\n';
      }
      values.close();
    }
  } else {
    throw std::runtime_error("gib c\n");
  }
  return 0;
}
