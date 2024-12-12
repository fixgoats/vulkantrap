#pragma once
#include "hack.hpp"
#include "mathhelpers.hpp"
#include "vk_mem_alloc.h"
#include <complex>
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>

typedef std::complex<double> c64;
typedef std::complex<float> c32;

constexpr float hbar = 6.582119569e-1;
constexpr float muB = 5.788e-2;
constexpr float echarge = 1e3;
constexpr float a0 = 0.01;
constexpr uint32_t nSpecConsts = 10;

struct SimConstants {
  uint32_t nElementsX;
  uint32_t nElementsY;
  uint32_t nElementsZ;
  uint32_t xGroupSize;
  uint32_t yGroupSize;
  float gamma;
  float Gamma;
  float R;
  float EXY;
  float dt;
  constexpr uint32_t X() const { return nElementsX / xGroupSize; }
  constexpr uint32_t Y() const { return nElementsY / yGroupSize; }
  constexpr bool validate() const {
    return (nElementsY % yGroupSize == 0) && (nElementsX % xGroupSize == 0);
  }
  constexpr uint32_t elementsTotal() const { return nElementsX * nElementsY; }
};

struct System {
  c32 psip;
  c32 psim;
  float np;
  float nm;
};

struct ParamConsts {
  float Omega0;
  float E0;
  float eta;
  float uX;
  float xi;
  float Gamma;
  float R;
  float gammadia;
  float gX;
  float p;
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

float rabi(float B, float Omega0);
constexpr float Epm(float B, float E0, float gX, float gammadia, bool sign) {
  return sign ? E0 - gX * muB * B + gammadia * B * B
              : E0 + gX * muB * B + gammadia * B * B;
}
float hopfield(float Omega, float E);
constexpr float uX0 = 0.01;
constexpr float Delta = 0.1;
constexpr c32 uX(float B_int, float B0, float gamma) {
  return uX0 * (c32{1, 0} + Delta / (B_int + c32{-B0, gamma}));
}
constexpr float Bint(float alphap, float alpham, float Gp, float Gm, float rhop,
                     float rhom, float np, float nm) {
  return alphap * rhop - alpham * rhom + Gp * np - Gm * nm;
}
constexpr float alpha(float X, c32 uX, float xi) {
  return sqrt(std::norm(xi * uX * X * X));
}
constexpr float G(float X, c32 uX) { return sqrt(std::norm(2.f * uX * X)); }

struct Params {
  float m_Ep;
  float m_Em;
  float m_Gp;
  float m_Gm;
  float m_alphap;
  float m_alpham;
  float m_Gammasp;
  float m_Gammasm;
  float m_Pp;
  float m_Pm;
  float m_PpdW;
  float m_PmdW;

  float E(float E0, float Omega) const;
  float G(float Omega, float uX, float E) const;

  float alpha(float Omega, float E, float xi, float uX) const;

  float P(float p, float Gamma, float gammaspm) const;
  Params(float B, ParamConsts p) {
    float omega = rabi(B, p.Omega0);
    float ep = Epm(B, p.E0, p.gX, p.gammadia, 1);
    float em = Epm(B, p.E0, p.gX, p.gammadia, 0);
    m_Ep = E(ep, omega);
    m_Em = E(em, omega);
    m_Gp = G(omega, p.uX, ep);
    m_Gm = G(omega, p.uX, em);
    m_alphap = alpha(omega, ep, p.xi, p.uX);
    m_alpham = alpha(omega, em, p.xi, p.uX);
    m_Gammasp = p.Gamma / 5. + p.eta * B;
    m_Gammasm = p.Gamma / 5. - p.eta * B;
    m_Pp = P(p.p, p.Gamma, m_Gammasm);
    m_Pm = P(p.p, p.Gamma, m_Gammasp);
    m_PpdW = 2 * m_Pp / p.Gamma;
    m_PmdW =
        2 * m_Pm /
        p.Gamma; // TODO: check if uploading PpdW and PmdW is actually any
                 // faster than calculating in shader. Optimizations performed
                 // during shader compilation might make this redundant.
  }
};

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
