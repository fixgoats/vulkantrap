#include "GPEsim.hpp"
#include <filesystem>

static const std::string appName{"Vulkan GPE Simulator"};
VulkanApp::VulkanApp(SimConstants sc) : params{sc} {
  vk::ApplicationInfo appInfo{appName.c_str(), 1, nullptr, 0,
                              VK_API_VERSION_1_3};
#ifndef DEBUG
  const std::vector<const char*> layers;
#else
  const std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
#endif // DEBUG
  vk::InstanceCreateInfo iCI(vk::InstanceCreateFlags(), &appInfo, layers, {});
  instance = vk::createInstance(iCI);
  pDevice = pickPhysicalDevice(instance);
  uint32_t computeQueueFamilyIndex = getComputeQueueFamilyIndex();
  float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo dQCI(vk::DeviceQueueCreateFlags(),
                                 computeQueueFamilyIndex, 1, &queuePriority);
  vk::DeviceCreateInfo dCI(vk::DeviceCreateFlags(), dQCI);
  device = pDevice.createDevice(dCI);
  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  computeQueueFamilyIndex);
  commandPool = device.createCommandPool(commandPoolCreateInfo);
  queue = device.getQueue(computeQueueFamilyIndex, 0);
  fence = device.createFence(vk::FenceCreateInfo());
  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.physicalDevice = pDevice;
  allocatorInfo.vulkanApiVersion = pDevice.getProperties().apiVersion;
  allocatorInfo.device = device;
  allocatorInfo.instance = instance;
  vmaCreateAllocator(&allocatorInfo, &allocator);
  uint32_t nElements = params.nElementsX * params.nElementsY;
  vk::BufferCreateInfo stagingBCI({}, 2 * nElements * sizeof(c32),
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  staging.allocate(allocator, allocCreateInfo, stagingBCI);
  vk::BufferCreateInfo systemBCI{vk::BufferCreateFlags(),
                                 nElements * sizeof(cvec2),
                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                     vk::BufferUsageFlagBits::eTransferDst |
                                     vk::BufferUsageFlagBits::eTransferSrc,
                                 vk::SharingMode::eExclusive,
                                 1,
                                 &computeQueueFamilyIndex};
  allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocCreateInfo.priority = 1.0f;
  computeBuffers.emplace_back(allocator, allocCreateInfo, systemBCI);
  std::vector<std::string> moduleNames = {"rk4sim.spv", "simplermodel.spv"};
  setupPipelines(moduleNames);
}

void VulkanApp::copyBuffers(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                            uint32_t bufferSize) {
  auto commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  commandBuffer.copyBuffer(srcBuffer, dstBuffer,
                           vk::BufferCopy(0, 0, bufferSize));
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  queue.submit(submitInfo, fence);
  queue.waitIdle();
  auto result = device.waitForFences(fence, true, -1);
  result = device.resetFences(1, &fence);
}

void VulkanApp::copyInBatches(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                              uint32_t batchSize, uint32_t numBatches) {
  auto commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);

  for (uint32_t i = 0; i < numBatches; i++) {
    commandBuffer.reset();
    commandBuffer.copyBuffer(
        srcBuffer, dstBuffer,
        vk::BufferCopy(i * batchSize, i * batchSize, batchSize));
  }
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  queue.submit(submitInfo, fence);
  auto result = device.waitForFences(fence, vk::True, -1);
  result = device.resetFences(1, &fence);
}

void VulkanApp::runSim(uint32_t n) {
  auto commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);

  for (uint32_t i = 0; i < params.times; i++) {
    appendPipeline(commandBuffer, n);
  }
  commandBuffer.end();

  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
  queue.submit(submitInfo, fence);
  queue.waitIdle();
  auto result = device.waitForFences(fence, vk::True, -1);
  result = device.resetFences(1, &fence);
}

void VulkanApp::initSystem() {
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.01, 0.01);
  cvec2* sStagingPtr = bit_cast<cvec2*>(staging.aInfo.pMappedData);
  for (uint32_t j = 0; j < params.nElementsY; j++) {
    for (uint32_t i = 0; i < params.nElementsX; i++) {
      sStagingPtr[j * params.nElementsX + i] =
          cvec2{{dis(gen), dis(gen)}, {dis(gen), dis(gen)}};
    }
  }
  copyBuffers(staging.buffer, computeBuffers[0].buffer,
              computeBuffers[0].aInfo.size);
}

void VulkanApp::appendPipeline(vk::CommandBuffer& cB, uint32_t i) {
  cB.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                     vk::PipelineStageFlagBits::eAllCommands, {},
                     fullMemoryBarrier, nullptr, nullptr);
  cB.bindPipeline(vk::PipelineBindPoint::eCompute, computePipelines[i]);
  cB.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0,
                        {descriptorSets[0]}, {});
  cB.dispatch(params.X(), params.Y(), 1);
}

void VulkanApp::initBuffers() { initSystem(); }

uint32_t VulkanApp::getComputeQueueFamilyIndex() {
  auto queueFamilyProps = pDevice.getQueueFamilyProperties();
  auto propIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(),
                   [](const vk::QueueFamilyProperties& prop) {
                     return prop.queueFlags & vk::QueueFlagBits::eCompute;
                   });
  return std::distance(queueFamilyProps.begin(), propIt);
}

void VulkanApp::rebuildPipelines(SimConstants p) {
  params = p;
  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(), dSL);
  pipelineLayout = device.createPipelineLayout(pLCI);
  pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());
  std::vector<vk::SpecializationMapEntry> bleh(nSpecConsts);
  for (uint32_t i = 0; i < nSpecConsts; i++) {
    bleh[i].constantID = i;
    bleh[i].offset = i * 4;
    bleh[i].size = 4;
  }
  vk::SpecializationInfo specInfo;
  specInfo.mapEntryCount = nSpecConsts;
  specInfo.pMapEntries = bleh.data();
  specInfo.dataSize = sizeof(SimConstants);
  specInfo.pData = &params;

  for (const auto& mod : modules) {
    vk::PipelineShaderStageCreateInfo cSCI(vk::PipelineShaderStageCreateFlags(),
                                           vk::ShaderStageFlagBits::eCompute,
                                           mod, "main", &specInfo);
    vk::ComputePipelineCreateInfo cPCI(vk::PipelineCreateFlags(), cSCI,
                                       pipelineLayout);
    auto result = device.createComputePipeline(pipelineCache, cPCI);
    assert(result.result == vk::Result::eSuccess);
    computePipelines.push_back(result.value);
  }
}

void VulkanApp::setupPipelines(std::vector<std::string> moduleNames) {
  for (const auto& name : moduleNames) {
    std::vector<uint32_t> shaderCode = readFile<uint32_t>(name);
    vk::ShaderModuleCreateInfo shaderMCI(vk::ShaderModuleCreateFlags(),
                                         shaderCode);
    modules.emplace_back(device.createShaderModule(shaderMCI));
  }
  std::vector<vk::DescriptorSetLayoutBinding> dSLBs;
  for (uint32_t i = 0; i < computeBuffers.size(); i++) {
    dSLBs.emplace_back(i, vk::DescriptorType::eStorageBuffer, 1,
                       vk::ShaderStageFlagBits::eCompute);
  }
  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  dSL = device.createDescriptorSetLayout(dSLCI);
  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(), dSL);
  pipelineLayout = device.createPipelineLayout(pLCI);
  pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());
  std::vector<vk::SpecializationMapEntry> bleh(nSpecConsts);
  for (uint32_t i = 0; i < nSpecConsts; i++) {
    bleh[i].constantID = i;
    bleh[i].offset = i * 4;
    bleh[i].size = 4;
  }
  vk::SpecializationInfo specInfo;
  specInfo.mapEntryCount = nSpecConsts;
  specInfo.pMapEntries = bleh.data();
  specInfo.dataSize = sizeof(SimConstants);
  specInfo.pData = &params;

  for (const auto& mod : modules) {
    vk::PipelineShaderStageCreateInfo cSCI(vk::PipelineShaderStageCreateFlags(),
                                           vk::ShaderStageFlagBits::eCompute,
                                           mod, "main", &specInfo);
    vk::ComputePipelineCreateInfo cPCI(vk::PipelineCreateFlags(), cSCI,
                                       pipelineLayout);
    auto result = device.createComputePipeline(pipelineCache, cPCI);
    assert(result.result == vk::Result::eSuccess);
    computePipelines.push_back(result.value);
  }

  vk::DescriptorPoolSize dPS(vk::DescriptorType::eStorageBuffer, 1);
  vk::DescriptorPoolCreateInfo dPCI(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, dPS);
  descriptorPool = device.createDescriptorPool(dPCI);
  vk::DescriptorSetAllocateInfo dSAI(descriptorPool, 1, &dSL);
  descriptorSets = device.allocateDescriptorSets(dSAI);
  descriptorSet = descriptorSets[0];
  std::vector<vk::DescriptorBufferInfo> dBIs;
  for (const auto& b : computeBuffers) {
    dBIs.emplace_back(b.buffer, 0, b.aInfo.size);
  }
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.emplace_back(descriptorSet, 0, 0, 1,
                                   vk::DescriptorType::eStorageBuffer, nullptr,
                                   &dBIs[0]);
  for (uint32_t i = 1; i < dBIs.size(); i++) {
    writeDescriptorSets.emplace_back(descriptorSet, i, 0, 1,
                                     vk::DescriptorType::eStorageBuffer,
                                     nullptr, &dBIs[i]);
  }
  device.updateDescriptorSets(writeDescriptorSets, {});
}

VulkanApp::~VulkanApp() {
  device.waitIdle();
  device.destroyFence(fence);
  for (auto& p : computePipelines) {
    device.destroyPipeline(p);
  }
  device.destroyPipelineCache(pipelineCache);
  device.destroyDescriptorPool(descriptorPool);
  for (auto& m : modules) {
    device.destroyShaderModule(m);
  }
  device.destroyPipelineLayout(pipelineLayout);
  device.destroyDescriptorSetLayout(dSL);
  staging.extirpate(allocator);
  for (auto& b : computeBuffers) {
    b.extirpate(allocator);
  }
  vmaDestroyAllocator(allocator);
  device.destroyCommandPool(commandPool);
  device.destroy();
  instance.destroy();
}

void VulkanApp::writeAllToCsv(std::string conffile) {
  auto Es = outputBuffer<uint32_t>(3);
  std::transform(Es.begin(), Es.end(), Es.begin(), [&](uint32_t x) {
    return static_cast<uint8_t>(fftshiftidx(x, 256));
  });
  auto system = outputBuffer<cvec2>(0);
  std::ofstream values;
  auto dir = std::format("data/{}", tstamp());
  std::filesystem::create_directories(dir);
  std::filesystem::copy(conffile, dir);
  values.open(std::format("{}/psip.csv", dir));
  for (uint32_t j = 0; j < params.nElementsY; ++j) {
    for (uint32_t i = 0; i < params.nElementsX; ++i) {
      values << std::format(" {}",
                            numfmt(system[j * params.nElementsX + i].psip));
    }
    values << '\n';
  }
  values.close();
  values.open(std::format("{}/psim.csv", dir));
  for (uint32_t j = 0; j < params.nElementsY; ++j) {
    for (uint32_t i = 0; i < params.nElementsX; ++i) {
      values << std::format(" {}",
                            numfmt(system[j * params.nElementsX + i].psim));
    }
    values << '\n';
  }
  values.close();
}
