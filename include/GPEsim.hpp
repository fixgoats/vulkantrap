#pragma once
#include "hack.hpp"
#include "vkhelpers.hpp"
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using std::bit_cast;

void appendOp(vk::CommandBuffer& b, Algorithm& a, u32 X, u32 Y = 1, u32 Z = 1);
void appendOpNoBarrier(vk::CommandBuffer& b, Algorithm& a, u32 X, u32 Y = 1,
                       u32 Z = 1);
struct VulkanApp {
  vk::Instance instance;
  vk::PhysicalDevice physicalDevice;
  vk::Device device;
  vk::Queue queue;
  vk::Fence fence;
  VmaAllocator allocator;
  vk::Buffer staging;
  VmaAllocation stagingAllocation;
  VmaAllocationInfo stagingInfo;
  uint32_t cQFI;
  vk::CommandPool commandPool;

  VulkanApp(size_t stagingSize);
  void copyBuffers(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                   uint32_t bufferSize);
  void copyInBatches(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                     uint32_t batchSize, uint32_t numBatches);

  vk::CommandBuffer beginRecord();
  void execute(vk::CommandBuffer& b);
  uint32_t getComputeQueueFamilyIndex();
  void writeToBuffer(MetaBuffer& buffer, const void* input, size_t size);
  template <class T>
  void writeToBuffer(MetaBuffer& buffer, const std::vector<T>& vec) {
    writeToBuffer(buffer, vec.data(), vec.size() * sizeof(T));
  }
  void writeFromBuffer(MetaBuffer& buffer, void* output, size_t size);
  template <class T>
  void writeFromBuffer(MetaBuffer& buffer, std::vector<T>& v) {
    writeFromBuffer(buffer, v.data(), v.size() * sizeof(T));
  }
  template <class T>
  void defaultInitBuffer(MetaBuffer& buffer, u32 nElements) {
    T* TStagingPtr = bit_cast<T*>(stagingInfo.pMappedData);
    for (u32 i = 0; i < nElements; i++) {
      TStagingPtr[i] = {};
    }
    copyBuffers(staging, buffer.buffer, nElements * sizeof(T));
  }
  template <typename T>
  MetaBuffer makeBuffer(uint32_t nElements) {
    vk::BufferCreateInfo bCI{vk::BufferCreateFlags(),
                             nElements * sizeof(T),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc,
                             vk::SharingMode::eExclusive,
                             1,
                             &cQFI};
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    allocCreateInfo.priority = 1.0f;
    return MetaBuffer{allocator, allocCreateInfo, bCI};
  }
  Algorithm makeAlgorithm(std::string spirvname,
                          std::vector<MetaBuffer*> buffers,
                          const u8* specConsts = nullptr,
                          const std::vector<u32>& specConstOffsets = {});
  ~VulkanApp();
};
