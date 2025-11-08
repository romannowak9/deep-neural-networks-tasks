#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

torch::Tensor radius_graph_cpu(torch::Tensor pos, double r, int64_t k = 0) {
    TORCH_CHECK(!pos.is_cuda(), "pos must be on CPU");
    TORCH_CHECK(pos.dim() == 2, "pos must be [N, D]");
    TORCH_CHECK(pos.scalar_type() == at::kFloat || pos.scalar_type() == at::kDouble,
                "pos must be float or double");

    const int64_t N = pos.size(0);
    const int64_t D = pos.size(1);
    const float r2 = static_cast<float>(r * r);

    auto pos_a = pos.accessor<float, 2>();
    std::vector<std::vector<std::pair<float,int64_t>>> neighbors(N);

    // ---- 1. lower-triangular radius edges
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            float dist2 = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                float diff = pos_a[i][d] - pos_a[j][d];
                dist2 += diff * diff;
            }
            if (dist2 < r2 && dist2 > 0.0f) {
                #pragma omp critical
                {
                    neighbors[i].emplace_back(dist2, j);
                }
            }
        }
    }

    // ---- 2. add self-loops before kNN
    for (int64_t i = 0; i < N; ++i)
        neighbors[i].emplace_back(0.0f, i);

    // ---- 3. optional k filtering (keep k smallest distances)
    std::vector<std::pair<int64_t,int64_t>> edges;
    edges.reserve(N * (k > 0 ? k : 8));

    for (int64_t i = 0; i < N; ++i) {
        auto &vec = neighbors[i];
        if (k > 0 && (int64_t)vec.size() > k) {
            std::partial_sort(vec.begin(), vec.begin()+k, vec.end(),
                              [](auto &a, auto &b){return a.first < b.first;});
            vec.resize(k);
        }
        for (auto &p : vec)
            edges.emplace_back(i, p.second);
    }

    // ---- 4. convert to tensor [E,2]
    const int64_t E = edges.size();
    auto edge_index = torch::empty({E,2}, torch::kLong);
    auto acc = edge_index.accessor<int64_t,2>();
    #pragma omp parallel for
    for (int64_t e = 0; e < E; ++e) {
        acc[e][0] = edges[e].first;
        acc[e][1] = edges[e].second;
    }
    return edge_index;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radius_graph_cpu", &radius_graph_cpu,
          "CPU radius_graph identical to Python",
          py::arg("pos"), py::arg("r"), py::arg("k") = 0);
}
