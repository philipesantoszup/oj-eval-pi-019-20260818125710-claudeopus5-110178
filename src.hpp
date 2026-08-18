#pragma once
#include "simulator.hpp"
#include <string>
#include <vector>

namespace sjtu {

namespace detail {

/*! \brief Largest divisor of `total` that does not exceed `limit`. */
inline size_t LargestDivisorAtMost(size_t total, size_t limit) {
  if (limit >= total) {
    return total;
  }
  for (size_t candidate = limit; candidate > 1; --candidate) {
    if (total % candidate == 0) {
      return candidate;
    }
  }
  return 1;
}

/*!
 * \brief Owner of the temporary matrices used by a single round.
 *
 * `MatrixMemoryAllocator` never frees the matrices it hands out, therefore the
 * temporaries of a finished round are deleted here.  Deletion happens only
 * after `GpuSimulator::Run()` has drained both instruction queues, so no
 * pending instruction can refer to them any more.
 */
class RoundScratch {
public:
  explicit RoundScratch(MatrixMemoryAllocator &allocator)
      : allocator_(allocator) {}

  RoundScratch(const RoundScratch &) = delete;
  RoundScratch &operator=(const RoundScratch &) = delete;

  Matrix *New(const std::string &name) {
    Matrix *matrix = allocator_.Allocate(name);
    owned_.push_back(matrix);
    return matrix;
  }

  /*! \brief Give up ownership of `matrix`, which outlives the current round. */
  void Disown(Matrix *matrix) {
    for (size_t i = 0; i < owned_.size(); ++i) {
      if (owned_[i] == matrix) {
        owned_.erase(owned_.begin() + i);
        return;
      }
    }
  }

  ~RoundScratch() {
    for (size_t i = 0; i < owned_.size(); ++i) {
      delete owned_[i];
    }
  }

private:
  MatrixMemoryAllocator &allocator_;
  std::vector<Matrix *> owned_;
};

} // namespace detail

/*
 * Round n (1-based) has to produce
 *
 *     Answer = Softmax(Q * K^T) * V ,   Q: (n, d),  K: (n, d),  V: (n, d)
 *
 * Two properties of the cost model shape the implementation.
 *
 * 1) MatMul costs 5 * size(lhs) * size(rhs).  A product that contracts over
 *    `len` elements thus costs 5 * len^2 per output element, which makes the
 *    rank-1 decomposition (outer products, len == 1) by far the cheapest way of
 *    multiplying:
 *
 *        Q * K^T = sum_{j<d} Q[:, j] (n,1) * K[:, j]^T (1,n)   ->  5 n^2 each
 *        P * V   = sum_{k<n} P[:, k] (n,1) * V[k, :]   (1,w)   ->  5 n w each
 *
 *    This turns the O(n^2 d^2) / O(n^3 d) naive cost into O(n^2 d).
 *
 * 2) Only the peak SRAM usage enters the score, HBM is free.  Consequently K, V
 *    and the queries are kept in HBM and are streamed through SRAM in single
 *    columns (Q, K) or in narrow feature slices (V, answer).  Streaming costs
 *    exactly the same IO as moving a whole matrix (300 cycles per element) but
 *    the resident SRAM stays at roughly 2 n^2 floats -- the score matrix and the
 *    probability matrix -- instead of the n*d floats of K or V.
 *
 * Instructions are issued so that the head of one queue never waits for an
 * instruction that was issued later on the other queue, which rules out
 * dead-locks while still letting IO and arithmetic overlap.
 */
void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t rounds = keys.size();
  if (rounds == 0) {
    return;
  }
  const size_t d = keys[0]->GetColumnNum(); // feature dimension
  // Number of features of V (and of the answer) processed at once.
  const size_t slice_width = detail::LargestDivisorAtMost(d, 8);

  Matrix *k_all = nullptr; // K, (n, d), in HBM
  Matrix *v_all = nullptr; // V, (n, d), in HBM
  bool own_k = false;      // whether K / V were allocated by us
  bool own_v = false;
  std::vector<Matrix *> retired; // freed at the start of the next round

  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < retired.size(); ++i) {
      delete retired[i];
    }
    retired.clear();

    Matrix *query = rater.GetNextQuery(); // (n, d), stays in HBM
    const size_t n = round + 1;
    detail::RoundScratch scratch(matrix_memory_allocator);

    /* ---- 1. append the key / value of this round --------------------- */
    if (k_all == nullptr) {
      k_all = keys[round];
      v_all = values[round];
    } else {
      Matrix *new_k = scratch.New("K");
      gpu_sim.Concat(k_all, keys[round], new_k, 0, kInGpuHbm);
      Matrix *new_v = scratch.New("V");
      gpu_sim.Concat(v_all, values[round], new_v, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(k_all);
      gpu_sim.ReleaseMatrix(keys[round]);
      gpu_sim.ReleaseMatrix(v_all);
      gpu_sim.ReleaseMatrix(values[round]);
      if (own_k) {
        retired.push_back(k_all);
      }
      if (own_v) {
        retired.push_back(v_all);
      }
      k_all = new_k;
      v_all = new_v;
      own_k = true;
      own_v = true;
      scratch.Disown(new_k);
      scratch.Disown(new_v);
    }

    /* ---- 2. scores = Q * K^T as a sum of d outer products ------------- */
    Matrix *scores = nullptr; // (n, n) in SRAM
    for (size_t j = 0; j < d; ++j) {
      Matrix *q_col = scratch.New("q_col");
      gpu_sim.GetColumn(query, j, q_col, kInGpuHbm); // (n, 1)
      gpu_sim.MoveMatrixToSharedMem(q_col);
      Matrix *k_col = scratch.New("k_col");
      gpu_sim.GetColumn(k_all, j, k_col, kInGpuHbm); // (n, 1)
      gpu_sim.MoveMatrixToSharedMem(k_col);
      gpu_sim.Transpose(k_col, kInSharedMemory); // (1, n)

      Matrix *update = scratch.New("score_update");
      gpu_sim.MatMul(q_col, k_col, update); // (n, n)
      gpu_sim.ReleaseMatrix(q_col);
      gpu_sim.ReleaseMatrix(k_col);
      if (scores == nullptr) {
        scores = update;
      } else {
        Matrix *accumulated = scratch.New("scores");
        gpu_sim.MatAdd(scores, update, accumulated);
        gpu_sim.ReleaseMatrix(scores);
        gpu_sim.ReleaseMatrix(update);
        scores = accumulated;
      }
    }

    /* ---- 3. row wise softmax ----------------------------------------- */
    Matrix *probs = nullptr; // (n, n) in SRAM
    for (size_t r = 0; r < n; ++r) {
      Matrix *row = scratch.New("score_row");
      gpu_sim.GetRow(scores, r, row, kInSharedMemory); // (1, n)
      Matrix *exp_row = scratch.New("exp_row");
      gpu_sim.MatExp(row, exp_row);
      gpu_sim.ReleaseMatrix(row);
      Matrix *row_sum = scratch.New("row_sum");
      gpu_sim.Sum(exp_row, row_sum); // (1, 1)
      Matrix *prob_row = scratch.New("prob_row");
      gpu_sim.MatDiv(exp_row, row_sum, prob_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);
      if (probs == nullptr) {
        probs = prob_row;
      } else {
        Matrix *stacked = scratch.New("probs");
        gpu_sim.Concat(probs, prob_row, stacked, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(probs);
        gpu_sim.ReleaseMatrix(prob_row);
        probs = stacked;
      }
    }
    gpu_sim.ReleaseMatrix(scores);

    /* ---- 4. answer = P * V, one narrow feature slice at a time ------- */
    Matrix *answer = nullptr; // (n, d) in HBM
    for (size_t first = 0; first < d; first += slice_width) {
      // Gather V[:, first : first + slice_width] into SRAM.
      Matrix *v_slice = nullptr; // (n, slice_width)
      for (size_t j = first; j < first + slice_width; ++j) {
        Matrix *v_col = scratch.New("v_col");
        gpu_sim.GetColumn(v_all, j, v_col, kInGpuHbm); // (n, 1)
        gpu_sim.MoveMatrixToSharedMem(v_col);
        if (v_slice == nullptr) {
          v_slice = v_col;
        } else {
          Matrix *wider = scratch.New("v_slice");
          gpu_sim.Concat(v_slice, v_col, wider, 1, kInSharedMemory);
          gpu_sim.ReleaseMatrix(v_slice);
          gpu_sim.ReleaseMatrix(v_col);
          v_slice = wider;
        }
      }

      // out_slice = sum_k P[:, k] * V_slice[k, :]
      Matrix *out_slice = nullptr; // (n, slice_width) in SRAM
      for (size_t k = 0; k < n; ++k) {
        Matrix *p_col = scratch.New("p_col");
        gpu_sim.GetColumn(probs, k, p_col, kInSharedMemory); // (n, 1)
        Matrix *v_row = scratch.New("v_row");
        gpu_sim.GetRow(v_slice, k, v_row, kInSharedMemory); // (1, w)
        Matrix *update = scratch.New("answer_update");
        gpu_sim.MatMul(p_col, v_row, update); // (n, w)
        gpu_sim.ReleaseMatrix(p_col);
        gpu_sim.ReleaseMatrix(v_row);
        if (out_slice == nullptr) {
          out_slice = update;
        } else {
          Matrix *accumulated = scratch.New("out_slice");
          gpu_sim.MatAdd(out_slice, update, accumulated);
          gpu_sim.ReleaseMatrix(out_slice);
          gpu_sim.ReleaseMatrix(update);
          out_slice = accumulated;
        }
      }
      gpu_sim.ReleaseMatrix(v_slice);
      gpu_sim.MoveMatrixToGpuHbm(out_slice);
      if (answer == nullptr) {
        answer = out_slice;
      } else {
        Matrix *wider = scratch.New("answer");
        gpu_sim.Concat(answer, out_slice, wider, 1, kInGpuHbm);
        gpu_sim.ReleaseMatrix(answer);
        gpu_sim.ReleaseMatrix(out_slice);
        answer = wider;
      }
    }
    gpu_sim.ReleaseMatrix(probs);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }

  for (size_t i = 0; i < retired.size(); ++i) {
    delete retired[i];
  }
  if (own_k) {
    delete k_all;
  }
  if (own_v) {
    delete v_all;
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
