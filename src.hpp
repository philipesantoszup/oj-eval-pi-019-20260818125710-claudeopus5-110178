#pragma once
#include "simulator.hpp"
#include <string>
#include <vector>

namespace sjtu {

namespace detail {

/*!
 * \brief Owner of the temporary matrices created inside a single round.
 *
 * The allocator of the framework never frees the matrices it hands out, so the
 * scratch matrices of a round are destroyed by this helper as soon as the round
 * has finished (i.e. after `GpuSimulator::Run`, when no instruction can refer
 * to them any more).  This keeps the host memory footprint constant.
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

  /*! \brief Stop owning `matrix` (it outlives the current round). */
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
 * Cost model remarks that drive the implementation:
 *
 *  * MatMul costs 5 * size(lhs) * size(rhs).  Therefore splitting the
 *    *reduction* axis into short pieces is a huge win: computing Q * K^T as
 *        sum_{j<d} Q[:, j] * (K[:, j])^T
 *    costs 5 * d * n^2 instead of 5 * d^2 * n^2 -- a factor d cheaper.  Only
 *    one column of Q and of K has to reside in SRAM at a time, so this also
 *    keeps the SRAM footprint minimal.  Single columns are extracted directly
 *    in HBM with GetColumn, which avoids moving whole matrices around.
 *
 *  * The score depends on the peak SRAM usage, hence
 *      - K and the queries stay in HBM and are streamed column by column,
 *      - the (n, n) score matrix is the only large intermediate,
 *      - the answer is produced row by row and immediately pushed back to HBM,
 *      - V is the single matrix that has to stay resident in SRAM; it grows by
 *        one row per round and is never transferred twice.
 *
 *  * Instructions of the two queues are emitted in an order that lets the IO
 *    engine work while the ALU is busy and never dead-locks (a queue head only
 *    ever waits for an instruction that was issued earlier on the other queue).
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

  Matrix *k_all = nullptr;  // K, shape (n, d), kept in HBM
  Matrix *v_all = nullptr;  // V, shape (n, d), kept in SRAM
  bool k_owned = false;     // whether we have to free K ourselves
  bool v_owned = false;
  std::vector<Matrix *> pending_delete; // freed at the beginning of next round

  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < pending_delete.size(); ++i) {
      delete pending_delete[i];
    }
    pending_delete.clear();

    Matrix *query = rater.GetNextQuery(); // (n, d), in HBM
    const size_t n = round + 1;
    detail::RoundScratch scratch(matrix_memory_allocator);

    /* ---- 1. append the key of this round to K and its value to V -------- */
    gpu_sim.MoveMatrixToSharedMem(values[round]); // V lives in SRAM
    if (k_all == nullptr) {
      k_all = keys[round];
      v_all = values[round];
    } else {
      Matrix *new_k = scratch.New("K");
      gpu_sim.Concat(k_all, keys[round], new_k, 0, kInGpuHbm);
      Matrix *new_v = scratch.New("V");
      gpu_sim.Concat(v_all, values[round], new_v, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(k_all);
      gpu_sim.ReleaseMatrix(keys[round]);
      gpu_sim.ReleaseMatrix(v_all);
      gpu_sim.ReleaseMatrix(values[round]);
      if (k_owned) {
        pending_delete.push_back(k_all);
      }
      if (v_owned) {
        pending_delete.push_back(v_all);
      }
      k_all = new_k;
      v_all = new_v;
      k_owned = true;
      v_owned = true;
      scratch.Disown(new_k);
      scratch.Disown(new_v);
    }

    /* ---- 2. scores = Q * K^T, accumulated column by column ------------- */
    Matrix *scores = nullptr; // (n, n) in SRAM
    for (size_t j = 0; j < d; ++j) {
      Matrix *q_col = scratch.New("q_col");
      gpu_sim.GetColumn(query, j, q_col, kInGpuHbm); // (n, 1)
      gpu_sim.MoveMatrixToSharedMem(q_col);
      Matrix *k_col = scratch.New("k_col");
      gpu_sim.GetColumn(k_all, j, k_col, kInGpuHbm); // (n, 1)
      gpu_sim.MoveMatrixToSharedMem(k_col);
      gpu_sim.Transpose(k_col, kInSharedMemory); // (1, n)

      Matrix *partial = scratch.New("partial_scores");
      gpu_sim.MatMul(q_col, k_col, partial); // (n, n)
      gpu_sim.ReleaseMatrix(q_col);
      gpu_sim.ReleaseMatrix(k_col);
      if (scores == nullptr) {
        scores = partial;
      } else {
        Matrix *accumulated = scratch.New("scores");
        gpu_sim.MatAdd(scores, partial, accumulated);
        gpu_sim.ReleaseMatrix(scores);
        gpu_sim.ReleaseMatrix(partial);
        scores = accumulated;
      }
    }

    /* ---- 3. row wise softmax combined with the multiplication by V ------ */
    Matrix *exps = scratch.New("exp_scores");
    gpu_sim.MatExp(scores, exps);
    gpu_sim.ReleaseMatrix(scores);

    Matrix *answer = nullptr; // (n, d) in HBM
    for (size_t r = 0; r < n; ++r) {
      Matrix *row = scratch.New("exp_row");
      gpu_sim.GetRow(exps, r, row, kInSharedMemory); // (1, n)
      Matrix *row_sum = scratch.New("row_sum");
      gpu_sim.Sum(row, row_sum); // (1, 1)
      Matrix *prob_row = scratch.New("prob_row");
      gpu_sim.MatDiv(row, row_sum, prob_row); // softmax of the row
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);

      Matrix *out_row = scratch.New("out_row");
      gpu_sim.MatMul(prob_row, v_all, out_row); // (1, d)
      gpu_sim.ReleaseMatrix(prob_row);
      gpu_sim.MoveMatrixToGpuHbm(out_row);
      if (answer == nullptr) {
        answer = out_row;
      } else {
        Matrix *taller = scratch.New("answer");
        gpu_sim.Concat(answer, out_row, taller, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(answer);
        gpu_sim.ReleaseMatrix(out_row);
        answer = taller;
      }
    }
    gpu_sim.ReleaseMatrix(exps);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }

  for (size_t i = 0; i < pending_delete.size(); ++i) {
    delete pending_delete[i];
  }
  if (k_owned) {
    delete k_all;
  }
  if (v_owned) {
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
