#include "typedefs.hpp"
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
  pallocator = &allocator;
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

MetaBuffer::~MetaBuffer() {
  vmaDestroyBuffer(*pallocator, static_cast<VkBuffer>(buffer), allocation);
}

Algorithm::Algorithm(vk::Device* device, std::vector<MetaBuffer*> buffers,
                     const std::vector<u32>& spirv, const u8* specConsts,
                     const std::vector<u32>& specConstOffsets) {
  p_Device = device;
  p_Buffer = buffers;
  vk::ShaderModuleCreateInfo shaderMCI(vk::ShaderModuleCreateFlags(), spirv);
  m_ShaderModule = device->createShaderModule(shaderMCI);
  std::vector<vk::DescriptorSetLayoutBinding> dSLBs;
  for (u32 i = 0; i < buffers.size(); i++) {
    dSLBs.emplace_back(i, vk::DescriptorType::eStorageBuffer, 1,
                       vk::ShaderStageFlagBits::eCompute);
  }
  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  m_DSL = device->createDescriptorSetLayout(dSLCI);
  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(), m_DSL);
  m_PipelineLayout = device->createPipelineLayout(pLCI);
  std::vector<vk::SpecializationMapEntry> specEntries(specConstOffsets.size());
  for (u32 i = 0; i < specEntries.size(); i++) {
    specEntries[i].constantID = i;
    specEntries[i].offset = i * 4;
    specEntries[i].size = 4;
  }
  vk::SpecializationInfo specInfo;
  specInfo.mapEntryCount = specEntries.size();
  specInfo.pMapEntries = specEntries.data();
  specInfo.dataSize = specConstOffsets.size() * 4;
  specInfo.pData = specConsts;

  vk::PipelineShaderStageCreateInfo cSCI(vk::PipelineShaderStageCreateFlags(),
                                         vk::ShaderStageFlagBits::eCompute,
                                         m_ShaderModule, "main", &specInfo);
  vk::ComputePipelineCreateInfo cPCI(vk::PipelineCreateFlags(), cSCI,
                                     m_PipelineLayout);
  auto result = device->createComputePipeline({}, cPCI);
  m_Pipeline = result.value;

  // This is probably not the most efficient way to do this, but I'm not going
  // to mess around with the descriptors after creation so the only overhead
  // should be memory, and I'm not going to make thousands of these so
  // it should be fine.
  vk::DescriptorPoolSize dPS(vk::DescriptorType::eStorageBuffer, 1);
  vk::DescriptorPoolCreateInfo dPCI(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, dPS);
  m_DescriptorPool = device->createDescriptorPool(dPCI);
  vk::DescriptorSetAllocateInfo dSAI(m_DescriptorPool, 1, &m_DSL);
  auto descriptorSets = device->allocateDescriptorSets(dSAI);
  m_DescriptorSet = descriptorSets[0];
  std::vector<vk::DescriptorBufferInfo> dBIs;
  for (const auto& b : buffers) {
    dBIs.emplace_back(b->buffer, 0, b->aInfo.size);
  }
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  for (uint32_t i = 0; i < dBIs.size(); i++) {
    writeDescriptorSets.emplace_back(m_DescriptorSet, i, 0, 1,
                                     vk::DescriptorType::eStorageBuffer,
                                     nullptr, &dBIs[i]);
  }
  device->updateDescriptorSets(writeDescriptorSets, {});
}

Algorithm::~Algorithm() {
  p_Device->destroyDescriptorSetLayout(m_DSL);
  p_Device->destroyDescriptorPool(m_DescriptorPool);
  p_Device->destroyShaderModule(m_ShaderModule);
  p_Device->destroyPipeline(m_Pipeline);
  p_Device->destroyPipelineLayout(m_PipelineLayout);
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
