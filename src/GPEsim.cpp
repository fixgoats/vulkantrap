#include "GPEsim.hpp"
#include "vkhelpers.hpp"
#include <filesystem>

static const std::string appName{"Vulkan GPE Simulator"};

VulkanApp::VulkanApp(size_t stagingSize) {
  vk::ApplicationInfo appInfo{appName.c_str(), 1, nullptr, 0,
                              VK_API_VERSION_1_3};
  const std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
  vk::InstanceCreateInfo iCI(vk::InstanceCreateFlags(), &appInfo, layers, {});
  instance = vk::createInstance(iCI);
  physicalDevice = pickPhysicalDevice(instance);
  cQFI = getComputeQueueFamilyIndex();
  float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo dQCI(vk::DeviceQueueCreateFlags(), cQFI, 1,
                                 &queuePriority);
  vk::DeviceCreateInfo dCI(vk::DeviceCreateFlags(), dQCI);
  device = physicalDevice.createDevice(dCI);
  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  cQFI);
  commandPool = device.createCommandPool(commandPoolCreateInfo);
  queue = device.getQueue(cQFI, 0);
  fence = device.createFence(vk::FenceCreateInfo());
  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.physicalDevice = physicalDevice;
  allocatorInfo.vulkanApiVersion = physicalDevice.getProperties().apiVersion;
  allocatorInfo.device = device;
  allocatorInfo.instance = instance;
  vmaCreateAllocator(&allocatorInfo, &allocator);
  vk::BufferCreateInfo stagingBCI({}, stagingSize,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  vmaCreateBuffer(allocator, bit_cast<VkBufferCreateInfo*>(&stagingBCI),
                  &allocCreateInfo, bit_cast<VkBuffer*>(&staging),
                  &stagingAllocation, &stagingInfo);
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
  auto result = device.waitForFences(fence, true, -1);
  result = device.resetFences(1, &fence);
  device.freeCommandBuffers(commandPool, commandBuffer);
}

void VulkanApp::writeToBuffer(MetaBuffer& buffer, const void* input,
                              size_t size) {
  memcpy(stagingInfo.pMappedData, input, size);
  copyBuffers(staging, buffer.buffer, size);
}

void VulkanApp::writeFromBuffer(MetaBuffer& buffer, void* input, size_t size) {
  copyBuffers(buffer.buffer, staging, size);
  memcpy(input, stagingInfo.pMappedData, size);
}

Algorithm VulkanApp::makeAlgorithm(std::string spirvname,
                                   std::vector<MetaBuffer*> buffers,
                                   const u8* specConsts,
                                   const std::vector<u32>& specConstOffsets) {
  const auto spirv = readFile<u32>(spirvname);
  return Algorithm(&device, buffers, spirv, specConsts, specConstOffsets);
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
  device.freeCommandBuffers(commandPool, commandBuffer);
}

void VulkanApp::execute(vk::CommandBuffer& b) {
  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &b);
  queue.submit(submitInfo, fence);
  auto result = device.waitForFences(fence, vk::True, -1);
  result = device.resetFences(1, &fence);
}

vk::CommandBuffer VulkanApp::beginRecord() {
  auto commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eSimultaneousUse);
  commandBuffer.begin(cBBI);

  return commandBuffer;
}

void appendOp(vk::CommandBuffer& b, Algorithm& a, u32 X, u32 Y, u32 Z) {
  b.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                    vk::PipelineStageFlagBits::eAllCommands, {},
                    fullMemoryBarrier, nullptr, nullptr);
  b.bindPipeline(vk::PipelineBindPoint::eCompute, a.m_Pipeline);
  b.bindDescriptorSets(vk::PipelineBindPoint::eCompute, a.m_PipelineLayout, 0,
                       a.m_DescriptorSet, nullptr);
  b.dispatch(X, Y, Z);
}

void appendOpNoBarrier(vk::CommandBuffer& b, Algorithm& a, u32 X, u32 Y,
                       u32 Z) {
  b.bindPipeline(vk::PipelineBindPoint::eCompute, a.m_Pipeline);
  b.bindDescriptorSets(vk::PipelineBindPoint::eCompute, a.m_PipelineLayout, 0,
                       a.m_DescriptorSet, nullptr);
  b.dispatch(X, Y, Z);
}

uint32_t VulkanApp::getComputeQueueFamilyIndex() {
  auto queueFamilyProps = physicalDevice.getQueueFamilyProperties();
  auto propIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(),
                   [](const vk::QueueFamilyProperties& prop) {
                     return prop.queueFlags & vk::QueueFlagBits::eCompute;
                   });
  return std::distance(queueFamilyProps.begin(), propIt);
}

VulkanApp::~VulkanApp() {
  device.waitIdle();
  device.destroyFence(fence);
  vmaDestroyBuffer(allocator, staging, stagingAllocation);
  vmaDestroyAllocator(allocator);
  device.destroyCommandPool(commandPool);
  device.destroy();
  instance.destroy();
}
