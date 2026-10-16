#pragma once

#include "bdg/graph.hpp"

namespace bdg::examples {

// SU(2), two site choices per interaction, one final connected output.
// Weights index a row-major 4n x 4n matrix diag(B_up, B_down).
Graph cos_su2_sublattice(unsigned order);
Graph diamond();

}  // namespace bdg::examples
