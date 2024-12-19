#pragma once
#include "hack.hpp"
#include "vkhelpers.hpp"
#include <bit>
#include <cstdint>
#include <random>

using std::bit_cast;

struct VulkanApp {
  vk::Instance instance;
  vk::PhysicalDevice pDevice;
  vk::Device device;
  vk::Queue queue;
  vk::Fence fence;
  VmaAllocator allocator;
  MetaBuffer staging;
  uint32_t nComplexBuffers;
  uint32_t nFloatBuffers;
  SimConstants params;
  std::vector<MetaBuffer> computeBuffers;
  std::vector<vk::ShaderModule> modules;
  vk::DescriptorSetLayout dSL;
  vk::PipelineLayout pipelineLayout;
  vk::PipelineCache pipelineCache;
  std::vector<vk::Pipeline> computePipelines;
  std::vector<vk::DescriptorSet> descriptorSets;
  vk::DescriptorSet descriptorSet;
  vk::DescriptorPool descriptorPool;
  vk::CommandPool commandPool;

  std::random_device rd;

  VulkanApp(SimConstants sc);
  void copyBuffers(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                   uint32_t bufferSize);
  void copyInBatches(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                     uint32_t batchSize, uint32_t numBatches);

  void runSim(uint32_t n);
  void initSystem();
  void s3();
  void appendPipeline(vk::CommandBuffer& cB, uint32_t i);
  void tests3();
  void initBuffers();
  uint32_t getComputeQueueFamilyIndex();
  void setupPipelines(std::vector<std::string> moduleNames);
  void rebuildPipelines(SimConstants p);
  void writeAllToCsv(std::string conffile);
  template <typename T>
  std::vector<T> outputBuffer(uint32_t n) {
    T* sStagingPtr = bit_cast<T*>(staging.aInfo.pMappedData);
    copyBuffers(computeBuffers[n].buffer, staging.buffer,
                computeBuffers[n].aInfo.size);
    std::vector<T> retVec(params.elementsTotal());
    memcpy(retVec.data(), sStagingPtr, computeBuffers[n].aInfo.size);
    return std::move(retVec);
  }

  ~VulkanApp();
};
