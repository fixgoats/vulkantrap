#include "rk4.h"

cvec2 f(cvec2 y, float x, float e) {
  cvec2 factor = {{x - (std::norm(y.x) + 2 * std::norm(y.y)),
                   -0.01f * (std::norm(y.x) + 2 * std::norm(y.y))},
                  {x - (std::norm(y.y) + 2 * std::norm(y.x)),
                   -0.01f * (std::norm(y.x) + 2 * std::norm(y.y))}};
  return y * factor + cvec2{c32{0, -e} * y.y, c32{0, -e} * y.x};
}

cvec2 rk4(cvec2 y, float dt, float x, float e) {
  cvec2 k1 = f(y, x, e);
  cvec2 k2 = f(y + (0.5f * dt) * k1, x, e);
  cvec2 k3 = f(y + (0.5f * dt) * k2, x, e);
  cvec2 k4 = f(y + dt * k3, x, e);
  return y + dt * (1.f / 6.f) * (k1 + (2.f * k2) + (2.f * k3) + k4);
}
