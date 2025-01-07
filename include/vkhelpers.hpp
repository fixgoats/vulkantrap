#pragma once
#include "hack.hpp"
#include "mathhelpers.hpp"
#include "typedefs.hpp"
#include "vk_mem_alloc.h"
#include <complex>
#include <cstdint>
#include <fstream>

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

std::string tstamp();

void saveToFile(std::string fname, const char* buf, size_t size);

struct SimConstants {
  uint32_t nElementsX;
  uint32_t nElementsY;
  uint32_t times;
  uint32_t xGroupSize;
  uint32_t yGroupSize;
  float pstart;
  float pend;
  float rstart;
  float rend;
  constexpr u32 X() const { return nElementsX / xGroupSize; }
  constexpr u32 Y() const { return nElementsY / yGroupSize; }
  constexpr bool validate() const {
    return (nElementsY % yGroupSize == 0) && (nElementsX % xGroupSize == 0);
  }
  constexpr uint32_t elementsTotal() const { return nElementsX * nElementsY; }
};
constexpr uint32_t nSpecConsts = sizeof(SimConstants) / 4;

std::ostream& operator<<(std::ostream& os, const SimConstants& obj);
std::ofstream& operator<<(std::ofstream& os, const SimConstants& obj);

const vk::MemoryBarrier fullMemoryBarrier(vk::AccessFlagBits::eShaderRead |
                                              vk::AccessFlagBits::eMemoryWrite,
                                          vk::AccessFlagBits::eMemoryRead |
                                              vk::AccessFlagBits::eMemoryWrite);

struct MetaBuffer {
  vk::Buffer buffer;
  VmaAllocator* pallocator = nullptr;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  MetaBuffer();
  MetaBuffer(VmaAllocator& allocator, VmaAllocationCreateInfo& allocCreateInfo,
             vk::BufferCreateInfo& BCI);
  // To call on default constructed metabuffer
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& allocCreateInfo,
                vk::BufferCreateInfo& BCI);
  ~MetaBuffer();
};

struct Algorithm {
  // Never owned
  vk::Device* p_Device;
  std::vector<MetaBuffer*> p_Buffer;
  // Owned
  vk::DescriptorSetLayout m_DSL;
  vk::DescriptorPool m_DescriptorPool;
  vk::DescriptorSet m_DescriptorSet;
  vk::ShaderModule m_ShaderModule;
  vk::PipelineLayout m_PipelineLayout;
  vk::Pipeline m_Pipeline;
  Algorithm(vk::Device* device, std::vector<MetaBuffer*> buffers,
            const std::vector<u32>& spirv, const u8* specConsts = nullptr,
            const std::vector<u32>& specConstOffsets = {});
  ~Algorithm();
};

template <class T>
void writeCsv(const std::string& filename, T* v, u32 nColumns, u32 nRows = 1,
              const std::vector<std::string>& heading = {}) {
  std::string out;
  if (heading.size()) {
    for (const auto& h : heading) {
      out = std::format("{} {}", out, h);
    }
    out = std::format("{}\n", out);
  }
  for (u32 j = 0; j < nRows; j++) {
    for (u32 i = 0; i < nColumns; i++) {
      out = std::format("{} {}", out, v[j * nColumns + i]);
    }
    out = std::format("{}\n", out);
  }
  std::ofstream of(filename);
  of << out;
  of.close();
}

std::vector<uint32_t> readFile(const std::string& filename);
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
