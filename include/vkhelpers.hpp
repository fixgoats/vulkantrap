#pragma once
#include "hack.hpp"
#include "mathhelpers.hpp"
#include "vk_mem_alloc.h"
#include <complex>
#include <cstdint>
#include <fstream>
#include <vulkan/vulkan_raii.hpp>

typedef std::complex<double> c64;
typedef std::complex<float> c32;

constexpr float hbar = 6.582119569e-1;
constexpr float muB = 5.788e-2;
constexpr float echarge = 1e3;
constexpr float a0 = 0.01;

template <typename T>
std::vector<T> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<T> buffer(fileSize / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
  file.close();
  return buffer;
}

template <class T>
constexpr auto numfmt(T x) {
  if constexpr (std::is_same_v<T, c32> or std::is_same_v<T, c64>) {
    return std::format("({}+{}j)", x.real(), x.imag());
  } else {
    return std::format("{}", x);
  }
}

std::string tstamp();

void saveToFile(std::string fname, const char* buf, size_t size);

struct SimConstants {
  uint32_t nElementsX;
  uint32_t nElementsY;
  uint32_t times;
  uint32_t xGroupSize;
  uint32_t yGroupSize;
  constexpr uint32_t X() const { return nElementsX / xGroupSize; }
  constexpr uint32_t Y() const { return nElementsY / yGroupSize; }
  constexpr bool validate() const {
    return (nElementsY % yGroupSize == 0) && (nElementsX % xGroupSize == 0);
  }
  constexpr uint32_t elementsTotal() const { return nElementsX * nElementsY; }
};
constexpr uint32_t nSpecConsts = sizeof(SimConstants) / 4;

std::ostream& operator<<(std::ostream& os, const SimConstants& obj);
std::ofstream& operator<<(std::ofstream& os, const SimConstants& obj);

struct cvec2 {
  c32 psip;
  c32 psim;
};

const vk::MemoryBarrier fullMemoryBarrier(vk::AccessFlagBits::eShaderRead |
                                              vk::AccessFlagBits::eMemoryWrite,
                                          vk::AccessFlagBits::eMemoryRead |
                                              vk::AccessFlagBits::eMemoryWrite);

struct MetaBuffer {
  vk::Buffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  MetaBuffer();
  MetaBuffer(VmaAllocator& allocator, VmaAllocationCreateInfo& allocCreateInfo,
             vk::BufferCreateInfo& BCI);
  // To call on default constructed metabuffer
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& allocCreateInfo,
                vk::BufferCreateInfo& BCI);
  void extirpate(VmaAllocator& allocator);
};

std::vector<uint32_t> readFile(const std::string& filename);
vk::raii::Instance makeInstance(const vk::raii::Context& context);
vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance,
                                      const int32_t desiredGPU = -1);

template <typename Func>
void oneTimeSubmit(const vk::Device& device, const vk::CommandPool& commandPool,
                   const vk::Queue& queue, const Func& func) {
  vk::CommandBuffer commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  commandBuffer.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  func(commandBuffer);
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  queue.submit(submitInfo, nullptr);
  queue.waitIdle();
}
