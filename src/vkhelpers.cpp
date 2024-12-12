#include <cstddef>
#include <format>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "vkhelpers.hpp"
#include <cstdint>
#include <iostream>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

void saveToFile(std::string fname, const char* buf, size_t size) {
  std::ofstream file(fname, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  file.write(buf, size);
  file.close();
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
