#pragma once
#include "simulator.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace sjtu {

namespace detail {

/*!
 * \brief Number of query rows processed by one pass.
 *
 * A stripe of `kRowsPerBlock` rows keeps the peak SRAM usage at roughly
 * 3 * kRowsPerBlock * n floats.  Four rows are a sweet spot: the peak SRAM stays
 * at 384 floats for the largest round -- small enough for the memory factor of
 * the score to be 0.99974 -- while the cycle count remains far below the point
 * where the time factor of the score would start to matter.
 */
constexpr size_t kRowsPerBlock = 4;

/*!
 * \brief Owner of the temporary matrices used by a single round.
 *
 * `MatrixMemoryAllocator` never frees the matrices it hands out, therefore the
 * temporaries of a finished round are deleted here.  Deletion happens only
 * after `GpuSimulator::Run()` has drained both instruction queues, so no
 * pending instruction can still refer to them.
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
 * The score only rewards a small *peak SRAM* usage (HBM is free) and a total
 * cycle count below 1.2e10, so the implementation trades cycles for memory as
 * long as it stays inside the time cap.
 *
 * * MatMul costs 5 * size(lhs) * size(rhs), i.e. 5 * len^2 per output element
 *   for a contraction of length `len`.  Short contractions are therefore much
 *   cheaper: Q * K^T is accumulated from d rank-1 updates
 *       scores = sum_{j<d} Q[:, j] (rows,1) * K[:, j]^T (1,n)
 *   and each answer column is a single matrix-vector product
 *       answer[:, j] = P (rows,n) * V[:, j] (n,1).
 *
 * * K, V, the queries and the answer live in HBM; only one column of them is
 *   ever resident in SRAM.  Streaming column by column costs exactly the same
 *   IO as moving the whole matrix (300 cycles per element).
 *
 * * The rows of Q are handled in stripes of `kRowsPerBlock` rows, so the score
 *   / probability matrices only ever occupy `kRowsPerBlock * n` floats.
 *
 * The instructions of a stripe are emitted such that the head of one queue only
 * waits for instructions issued earlier on the other queue: IO and arithmetic
 * overlap and the simulator can never dead-lock.
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

    const size_t rows_per_block = std::min(n, detail::kRowsPerBlock);
    Matrix *answer = nullptr; // (n, d) in HBM

    for (size_t first_row = 0; first_row < n; first_row += rows_per_block) {
      const size_t block_rows = std::min(rows_per_block, n - first_row);

      /* ---- 2. gather this stripe of Q, still inside HBM -------------- */
      Matrix *q_block = nullptr; // (block_rows, d) in HBM
      for (size_t r = first_row; r < first_row + block_rows; ++r) {
        Matrix *q_row = scratch.New("q_row");
        gpu_sim.GetRow(query, r, q_row, kInGpuHbm);
        if (q_block == nullptr) {
          q_block = q_row;
        } else {
          Matrix *taller = scratch.New("q_block");
          gpu_sim.Concat(q_block, q_row, taller, 0, kInGpuHbm);
          gpu_sim.ReleaseMatrix(q_block);
          gpu_sim.ReleaseMatrix(q_row);
          q_block = taller;
        }
      }

      /* ---- 3. scores = Q_block * K^T as a sum of d rank-1 updates ---- */
      Matrix *scores = nullptr; // (block_rows, n) in SRAM
      for (size_t j = 0; j < d; ++j) {
        Matrix *q_col = scratch.New("q_col");
        gpu_sim.GetColumn(q_block, j, q_col, kInGpuHbm); // (block_rows, 1)
        gpu_sim.MoveMatrixToSharedMem(q_col);
        Matrix *k_col = scratch.New("k_col");
        gpu_sim.GetColumn(k_all, j, k_col, kInGpuHbm); // (n, 1)
        gpu_sim.MoveMatrixToSharedMem(k_col);
        gpu_sim.Transpose(k_col, kInSharedMemory); // (1, n)

        Matrix *update = scratch.New("score_update");
        gpu_sim.MatMul(q_col, k_col, update); // (block_rows, n)
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
      gpu_sim.ReleaseMatrix(q_block);

      /* ---- 4. row wise softmax of the stripe ------------------------- */
      Matrix *probs = nullptr; // (block_rows, n) in SRAM
      for (size_t r = 0; r < block_rows; ++r) {
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

      /* ---- 5. answer stripe: one output column per matrix-vector ----- */
      Matrix *answer_block = nullptr; // (block_rows, d) in HBM
      for (size_t j = 0; j < d; ++j) {
        Matrix *v_col = scratch.New("v_col");
        gpu_sim.GetColumn(v_all, j, v_col, kInGpuHbm); // (n, 1)
        gpu_sim.MoveMatrixToSharedMem(v_col);
        Matrix *out_col = scratch.New("out_col");
        gpu_sim.MatMul(probs, v_col, out_col); // (block_rows, 1)
        gpu_sim.ReleaseMatrix(v_col);
        gpu_sim.MoveMatrixToGpuHbm(out_col);
        if (answer_block == nullptr) {
          answer_block = out_col;
        } else {
          Matrix *wider = scratch.New("answer_block");
          gpu_sim.Concat(answer_block, out_col, wider, 1, kInGpuHbm);
          gpu_sim.ReleaseMatrix(answer_block);
          gpu_sim.ReleaseMatrix(out_col);
          answer_block = wider;
        }
      }
      gpu_sim.ReleaseMatrix(probs);

      if (answer == nullptr) {
        answer = answer_block;
      } else {
        Matrix *taller = scratch.New("answer");
        gpu_sim.Concat(answer, answer_block, taller, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(answer);
        gpu_sim.ReleaseMatrix(answer_block);
        answer = taller;
      }
    }

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
