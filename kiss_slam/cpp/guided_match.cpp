// Guided matching of panorama keypoints (#087): for every keypoint of the earlier panorama, the best and second-best descriptor
// distances among the later panorama's keypoints within +-wx columns (wrap-around at the panorama width) and +-wy rows of the same
// pixel.  The later keypoints are bucketed in a grid of cells at least one window wide, so each query visits 3 x 3 cells; distances
// are plain L2 on float descriptors (vectorised by the compiler), queries in parallel (OpenMP).  The ratio test is applied in Python.
//
// Build: scripts/build_guided_match.sh  ->  kiss_slam/_guided_match<EXT_SUFFIX>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;
using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::tuple guided_match(farray p1, farray d1, farray p2, farray d2, float width, float wx, float wy) {
    const ssize_t n = p1.shape(0), m = p2.shape(0), dim = d1.shape(1);
    if (d2.shape(1) != dim) throw std::runtime_error("descriptor sizes differ");
    const float *P1 = p1.data(), *P2 = p2.data(), *D1 = d1.data(), *D2 = d2.data();

    const int nx = std::max(3, static_cast<int>(width / wx));              // >= 3 so the 3 x 3 neighbourhood has distinct columns
    const float cw = width / nx;                                            // cell width >= wx when width / wx >= 3
    float ymax = 0.f;
    for (ssize_t j = 0; j < m; ++j) ymax = std::max(ymax, P2[2 * j + 1]);
    for (ssize_t i = 0; i < n; ++i) ymax = std::max(ymax, P1[2 * i + 1]);
    const int ny = std::max(1, static_cast<int>(std::ceil((ymax + 1.f) / wy)));
    auto cx_of = [&](float x) { int c = static_cast<int>(std::floor(x / cw)) % nx; return c < 0 ? c + nx : c; };
    auto cy_of = [&](float y) { return std::min(std::max(static_cast<int>(y / wy), 0), ny - 1); };

    std::vector<std::vector<int>> cells(static_cast<size_t>(nx) * ny);
    for (ssize_t j = 0; j < m; ++j) cells[cy_of(P2[2 * j + 1]) * nx + cx_of(P2[2 * j])].push_back(static_cast<int>(j));

    py::array_t<int> idx(n);
    py::array_t<float> best(n), second(n);
    int *I = idx.mutable_data();
    float *B = best.mutable_data(), *S = second.mutable_data();
    const float inf = std::numeric_limits<float>::infinity();

#pragma omp parallel for schedule(dynamic, 64)
    for (ssize_t i = 0; i < n; ++i) {
        const float x = P1[2 * i], y = P1[2 * i + 1];
        const float *a = D1 + i * dim;
        float b1 = inf, b2 = inf; int j1 = -1;
        const int cx = cx_of(x), cy = cy_of(y);
        for (int dy = -1; dy <= 1; ++dy) {
            const int yy = cy + dy;
            if (yy < 0 || yy >= ny) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int xx = (cx + dx + nx) % nx;
                for (int j : cells[yy * nx + xx]) {
                    float ddx = std::fabs(P2[2 * j] - x);
                    ddx = std::min(ddx, width - ddx);
                    if (ddx > wx || std::fabs(P2[2 * j + 1] - y) > wy) continue;
                    const float *b = D2 + static_cast<ssize_t>(j) * dim;
                    float s = 0.f;
                    for (ssize_t k = 0; k < dim; ++k) { const float t = a[k] - b[k]; s += t * t; }
                    if (s < b1) { b2 = b1; b1 = s; j1 = j; } else if (s < b2) { b2 = s; }
                }
            }
        }
        I[i] = j1; B[i] = std::sqrt(b1); S[i] = std::sqrt(b2);
    }
    return py::make_tuple(idx, best, second);
}

// The same for binary descriptors (ORB, #088): Hamming distance by popcount over 8-byte words.
using barray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
py::tuple guided_match_hamming(farray p1, barray d1, farray p2, barray d2, float width, float wx, float wy) {
    const ssize_t n = p1.shape(0), m = p2.shape(0), nb = d1.shape(1);
    if (d2.shape(1) != nb || nb % 8 != 0) throw std::runtime_error("descriptor sizes differ or are not a multiple of 8 bytes");
    const ssize_t words = nb / 8;
    const float *P1 = p1.data(), *P2 = p2.data();
    const uint64_t *D1 = reinterpret_cast<const uint64_t *>(d1.data()), *D2 = reinterpret_cast<const uint64_t *>(d2.data());
    const int nx = std::max(3, static_cast<int>(width / wx));
    const float cw = width / nx;
    float ymax = 0.f;
    for (ssize_t j = 0; j < m; ++j) ymax = std::max(ymax, P2[2 * j + 1]);
    for (ssize_t i = 0; i < n; ++i) ymax = std::max(ymax, P1[2 * i + 1]);
    const int ny = std::max(1, static_cast<int>(std::ceil((ymax + 1.f) / wy)));
    auto cx_of = [&](float x) { int c = static_cast<int>(std::floor(x / cw)) % nx; return c < 0 ? c + nx : c; };
    auto cy_of = [&](float y) { return std::min(std::max(static_cast<int>(y / wy), 0), ny - 1); };
    std::vector<std::vector<int>> cells(static_cast<size_t>(nx) * ny);
    for (ssize_t j = 0; j < m; ++j) cells[cy_of(P2[2 * j + 1]) * nx + cx_of(P2[2 * j])].push_back(static_cast<int>(j));
    py::array_t<int> idx(n);
    py::array_t<float> best(n), second(n);
    int *I = idx.mutable_data();
    float *B = best.mutable_data(), *S = second.mutable_data();
    const float inf = std::numeric_limits<float>::infinity();
#pragma omp parallel for schedule(dynamic, 64)
    for (ssize_t i = 0; i < n; ++i) {
        const float x = P1[2 * i], y = P1[2 * i + 1];
        const uint64_t *a = D1 + i * words;
        int b1 = std::numeric_limits<int>::max(), b2 = b1, j1 = -1;
        const int cx = cx_of(x), cy = cy_of(y);
        for (int dy = -1; dy <= 1; ++dy) {
            const int yy = cy + dy;
            if (yy < 0 || yy >= ny) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int xx = (cx + dx + nx) % nx;
                for (int j : cells[yy * nx + xx]) {
                    float ddx = std::fabs(P2[2 * j] - x);
                    ddx = std::min(ddx, width - ddx);
                    if (ddx > wx || std::fabs(P2[2 * j + 1] - y) > wy) continue;
                    const uint64_t *b = D2 + static_cast<ssize_t>(j) * words;
                    int s = 0;
                    for (ssize_t k = 0; k < words; ++k) s += __builtin_popcountll(a[k] ^ b[k]);
                    if (s < b1) { b2 = b1; b1 = s; j1 = j; } else if (s < b2) { b2 = s; }
                }
            }
        }
        I[i] = j1;
        B[i] = j1 < 0 ? inf : static_cast<float>(b1);
        S[i] = b2 == std::numeric_limits<int>::max() ? inf : static_cast<float>(b2);
    }
    return py::make_tuple(idx, best, second);
}

PYBIND11_MODULE(_guided_match, m) {
    m.def("guided_match", &guided_match, "best / second-best descriptor distance within a pixel window (#087)",
          py::arg("p1"), py::arg("d1"), py::arg("p2"), py::arg("d2"), py::arg("width"), py::arg("wx"), py::arg("wy"));
    m.def("guided_match_hamming", &guided_match_hamming, "the same for binary descriptors, Hamming distance (#088)",
          py::arg("p1"), py::arg("d1"), py::arg("p2"), py::arg("d2"), py::arg("width"), py::arg("wx"), py::arg("wy"));
}
