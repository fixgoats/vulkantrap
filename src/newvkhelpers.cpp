#include <cstddef>
#define VMA_IMPLEMENTATION
#include "newvkhelpers.hpp"
#include "vk_mem_alloc.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

std::vector<uint32_t> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> buffer(fileSize / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
  file.close();
  return buffer;
}

float rabi(float B, float Omega0) {
  return (Omega0 / sqrt(2.)) *
         sqrt(sqrt(1 + 1.5 * square(echarge * square(a0) * B / hbar)) + 1);
}

float hopfield(float Omega, float E) {
  return 0.5 * (1 - (E / sqrt(E * E + square(2 * Omega))));
}

MetaBuffer::MetaBuffer() {
  buffer = vk::Buffer{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
}

MetaBuffer::MetaBuffer(VmaAllocator& allocator,
                       VmaAllocationCreateInfo& allocCreateInfo,
                       vk::BufferCreateInfo& BCI) {
  buffer = vk::Buffer{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
  vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&BCI),
                  &allocCreateInfo, reinterpret_cast<VkBuffer*>(&buffer),
                  &allocation, &aInfo);
}

void MetaBuffer::allocate(VmaAllocator& allocator,
                          VmaAllocationCreateInfo& allocCreateInfo,
                          vk::BufferCreateInfo& BCI) {
  vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&BCI),
                  &allocCreateInfo, reinterpret_cast<VkBuffer*>(&buffer),
                  &allocation, &aInfo);
}

void MetaBuffer::extirpate(VmaAllocator& allocator) {
  vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer), allocation);
}

float Params::E(float E, float Omega) const {
  return (0.5 * E) - (0.5 * sqrt(square(2 * Omega) + E * E));
}

float Params::G(float Omega, float uX, float E) const {
  return 2 * uX * hopfield(Omega, E);
}

float Params::alpha(float Omega, float E, float xi, float uX) const {
  return (xi * uX) * square(hopfield(Omega, E));
}

float Params::P(float p, float Gamma, float gammaspm) const {
  return p / (9. * Gamma / 10.) * (Gamma / 4 + gammaspm);
}

vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance,
                                      const int32_t desiredGPU) {
  // check if there are GPUs that support Vulkan and "intelligently" select
  // one. Prioritises discrete GPUs, and after that VRAM size.
  std::vector<vk::PhysicalDevice> pDevices =
      instance.enumeratePhysicalDevices();
  uint32_t nDevices = pDevices.size();

  // shortcut if there's only one device available.
  if (nDevices == 1) {
    return pDevices[0];
  }
  // Try to select desired GPU if specified.
  if (desiredGPU > -1) {
    if (desiredGPU < static_cast<int32_t>(nDevices)) {
      return pDevices[desiredGPU];
    } else {
      std::cout << "Device not available\n";
    }
  }

  std::vector<uint32_t> discrete; // the indices of the available discrete gpus
  std::vector<uint64_t> vram(nDevices);
  for (uint32_t i = 0; i < nDevices; i++) {
    if (pDevices[i].getProperties().deviceType ==
        vk::PhysicalDeviceType::eDiscreteGpu) {
      discrete.push_back(i);
    }

    auto heaps = pDevices[i].getMemoryProperties().memoryHeaps;
    for (const auto& heap : heaps) {
      if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
        vram[i] = heap.size;
      }
    }
  }

  // only consider discrete gpus if available:
  if (discrete.size() > 0) {
    if (discrete.size() == 1) {
      return pDevices[discrete[0]];
    } else {
      uint32_t max = 0;
      uint32_t selectedGPU = 0;
      for (const auto& index : discrete) {
        if (vram[index] > max) {
          max = vram[index];
          selectedGPU = index;
        }
      }
      return pDevices[selectedGPU];
    }
  } else {
    uint32_t max = 0;
    uint32_t selectedGPU = 0;
    for (uint32_t i = 0; i < nDevices; i++) {
      if (vram[i] > max) {
        max = vram[i];
        selectedGPU = i;
      }
    }
    return pDevices[selectedGPU];
  }
}
