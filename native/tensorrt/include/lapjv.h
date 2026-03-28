#pragma once

#include <limits>
#include <vector>

namespace oc_sort {
double execLapjv(const std::vector<std::vector<float>> &cost,
                 std::vector<int> &rowsol, std::vector<int> &colsol,
                 bool extend_cost = true,
                 float cost_limit = std::numeric_limits<float>::max(),
                 bool return_cost = true);
} // namespace oc_sort