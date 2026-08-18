# Problem 019 - Mini-Aidiv-N: GPU Memory Optimization for Attention

**ACMOJ Problem ID**: 2782

## Table of Contents

- [Problem 019 - Mini-Aidiv-N: GPU Memory Optimization for Attention](#problem-019---mini-aidiv-n-gpu-memory-optimization-for-attention)
  - [Table of Contents](#table-of-contents)
  - [Problem Description](#problem-description)
  - [Background: GPU Memory Hierarchy](#background-gpu-memory-hierarchy)
  - [Background: Attention Computation](#background-attention-computation)
  - [GPU Simulator Architecture](#gpu-simulator-architecture)
  - [Instruction Set](#instruction-set)
    - [Calculation Instructions](#calculation-instructions)
    - [IO Instructions](#io-instructions)
  - [Input Format](#input-format)
  - [Output Format](#output-format)
  - [FAQ](#faq)
  - [Important Notes](#important-notes)
  - [Test Data](#test-data)
  - [Data Range](#data-range)
    - [Performance Scoring Formula](#performance-scoring-formula)
    - [Grade Composition](#grade-composition)
  - [Per-Testcase Resource Limits](#per-testcase-resource-limits)
  - [Submission Requirements](#submission-requirements)
    - [File Submission](#file-submission)
    - [Submission Guidelines](#submission-guidelines)
    - [What to Implement](#what-to-implement)
    - [Evaluation Notes](#evaluation-notes)
    - [Academic Integrity](#academic-integrity)

## Problem Description

> You may say I'm a dreamer, but **a1** is a brand new open GPU computing game independently developed by Parsifal and Yuchuan. The game takes place in a fantasy world called **Mini-Aidiv-N**, where those chosen by the gods will be granted the power of **parallel computing**, guiding the forces of optimization. You will play a mysterious character named **R.X.Huang**, encountering components with unique personalities and abilities during your free journey, working with them to complete computational tasks while saving as much time as possible — and gradually uncovering the truth about computation.

## Background: GPU Memory Hierarchy

Generally speaking, GPU memory hierarchy is as follows:

![GPU Memory Hierarchy](GPU%20Memory%20Hierarchy.png)

Simply put, the higher the level, the smaller the storage space, but the faster the transfer speed. When GPU computation is complete, our computation results are all stored in the **highest level**. Meanwhile, to allow data to enter the GPU for computation, in most cases, we store all operands in the **highest level** (in this problem, it's SRAM).

Generally speaking, GPU memory hierarchy is as shown above:
- SRAM is the highest level. Its size is smallest, but transfer speed is fastest.
- HBM is the second highest level. Its size is medium, and transfer speed is relatively fast.
- DRAM is the lowest level. Its size can be very large, but transfer speed is relatively slow.

In this problem, since **a1** is an open GPU computing game, we can assume our HBM is **infinitely large**, so we don't need to consider the DRAM level. That is, our memory hierarchy only consists of SRAM and HBM, where SRAM is the highest level.

## Background: Attention Computation

![Attention Computation](Attention%20Computation.png)

The above figure shows a common computation method for Attention. Specifically, given three matrices $Q, K, V$, we use the method shown in the right figure to compute their attention. The process is as follows:

1. Given $Q, K$, compute $\text{MatMul}(Q, K^T) = QK^T$
2. Then perform a $\text{Scale}$ operation on it. In this problem, you **don't need to consider this operation**.
3. Then, perform a $\text{Mask}$ operation on the result of $\text{Scale}$. In this problem, you **don't need to consider this operation**.
4. For the result from previous steps, we perform a $\text{Softmax}$ operation on the **rows** of the matrix.
5. Multiply the result of $\text{Softmax}$ with matrix $V$. In this problem, it's $\text{MatMul}(\text{Softmax}(\text{MatMul}(Q, K^T)),V)$

Your task is to accept given matrices, complete specified operations, and return appropriate results.

The definition of $\text{Softmax}$ is: $\text{Softmax}(\vec A):\text{Softmax}(A[i]) = \frac{\text{e}^{A[i]}}{\sum_{A[j]\in A}\text{e}^{A[j]}}$ 

## GPU Simulator Architecture

In the Simulator, a matrix's location can be in one of three states: in `hbm`, in `sram`, or `released`. Most computational work requires you to move data to `sram` for processing, and return the answer to `hbm`. We provide a variety of instructions. You can understand the function usage by reading the table below, code comments, or asking TAs (try not to ask too much).

## Instruction Set

In this problem, you have **two types** of instructions: `Calculation` instructions and `IO` instructions. Instructions within the same type execute serially, but these two different types of instructions execute **in parallel**.

Each instruction has a certain execution time. We assume $M_1\in \mathbb{R}^{m\times n}, M_2\in \mathbb{R}^{n\times q}, M_3\in \mathbb{R}^{m\times k}$. The specific execution time for each instruction is as follows (for specific meanings of each instruction, see comments):

### Calculation Instructions

#### SRAM Calculation Instructions (operands must be in SRAM, results are also stored in SRAM)

| Function Name | Time Cost (clock) | Memory Strategy |
| - | - | - |
| $\text{MatAdd}(M_1,M_1)$ | $mn$ | Allocate new SRAM |
| $\text{MatSub}(M_1,M_1)$ | $mn$ | Allocate new SRAM |
| $\text{MatMul}(M_1,M_2)$ | $5mn^2q$ | Allocate new SRAM |
| $\text{MatDiv}(M_1, [k])$ | $16mn$| Allocate new SRAM |
| $\text{AddBias}(M_1, [k])$ | $mn$| Allocate new SRAM |
| $\text{Sum}(M_1)$ | $mn$ | Allocate new SRAM |
| $\text{MatExp}(M_1)$|$30mn$| Allocate new SRAM |

#### General Calculation Instructions (operands can be in HBM, but speed may differ in different memory regions)

| Function Name | Time Cost (clock) | Memory Strategy |
| - | - | - |
| $\text{Concat}(M_1, M_3)$ | $25m(n+k)$ (HBM), $m(n+k)$ (SRAM)| Allocate new space of same type as operands |
| $\text{GetRow}(M_1, k)$ | $25n$ (HBM), $n$ (SRAM) | Allocate new space of same type as operands |
| $\text{GetColumn}(M_1, k)$|$25m$ (HBM), $m$ (SRAM)| Allocate new space of same type as operands |
| $\text{Copy}(M_1)$|$25mn$ (HBM), $mn$ (SRAM)| Allocate new space of same type as operands |
| $\text{ReleaseMatrix}(M_1)$| $10$ | Release space |
| $\text{Reshape}(M_1, rows)$ | $1$ | In-place modification |
| $\text{Transpose}(M_1)$| $25mn$ (HBM), $mn$ (SRAM) | In-place modification |

### IO Instructions

| Function Name | Time Cost (clock) | Memory Strategy |
| - | - | - |
| $\text{MoveMatrixToGpuHbm}(M_1)$| $300mn$| Allocate new HBM, release original SRAM |
| $\text{MoveMatrixToSharedMem}(M_1)$| $300mn$| Allocate new SRAM, release original HBM |

## Input Format

In this problem, you need to complete the provided function to implement a `Calculate` function for the `GPU Simulator`. The code is as follows:

```cpp
void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    /*
     *
     *
     *
     *
     *
     *
     * YOUR CODE HERE
     *
     *
     *
     *
     *
     *
     */

    /*********************  End of your code *********************/
    gpu_sim.Run(false, &matrix_memory_allocator);
	//rater.CommitAnswer(YOUR_ANSWER_MATRIX);(commit after running the simulator)
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}
```

When submitting, please submit the entire `src.hpp` file.

The incoming `keys, values, current_query` etc. are all stored in `hbm`.

Specifically, our program will run `n` rounds. In round `i`, the workflow is as follows:
- Request the matrix `Q` for this round from the Rater.
- Perform the **Attention** operation mentioned above with the first `i` groups of `K` (keys) and `V` (values).
- Sum the results from `i` groups and move the result to **HBM**.
- Run `gpu_sim->Run();` and submit your answer to the `rater`.

**Please note**
- **Round indexing in this problem is 1-based** (i.e., the first round is round 1, not round 0).
- `K[0], K[1], ... K[i]`, `V[0], V[1], ...V[i]` are all matrices with `shape=[1, d]`, and `Q` is a matrix with `shape=[i+1, d]`.
- Specifically, you need to complete the following computational task:

![Computation Task](Attention%20Computation.png)

- Finally, perform `rater.CommitAnswer` on the `Answer` matrix shown above.

## Output Format

Your program should output the result through the rater interface as described above.

## FAQ

#### Can I specify that the result of an instruction should be stored in the operand of that instruction?

No. Although we know this is a very common optimization strategy, if you do this in this problem, it may lead to various undefined results.

#### I specified only three instructions. The first requires moving data A from HBM to SRAM, the second requires data A to participate in computation, and the third instruction requires data originally in SRAM to participate in computation. How will the simulator work in this case?

Because **instructions of the same type** are serial internally, the simulator will wait for you to transfer data `A` to SRAM before starting to execute the second instruction, and then execute the third instruction. It will not skip the second instruction and execute the third instruction early.

#### I need to transfer the result of an instruction operation to HBM. How will the simulator work in this case?

Same as above. The simulator will wait for you to complete the computation before transferring.

#### Some Calculation Instructions require providing a Position parameter. What does this mean?

`Position` indicates which component you want this instruction to run in. If the position specified by `Position` doesn't match the position of the operands, the calculation queue will wait until the operand position meets the requirements. The operation result will also be saved in that component.

## Important Notes

- Do not output any other content to the standard output stream!
- Do not assign values to the `Matrix` class. All assignment operations for Matrix should be completed by `gpu_simulator`.
- Do not let the same `Matrix` serve as the result object for multiple instructions. Each `Matrix *` passed into instructions should be in an unassigned state.
- Please do not attempt to exploit vulnerabilities in the evaluation program. This exam is not a CTF problem.
- Before running `CommitAnswer`, please let the Simulator `Run`.
- If you cannot complete everything, you can submit partial answers and then submit any matrix in HBM afterward. This should also earn some points.

If you violate the above notes, your code will be treated as `void`.

## Test Data

Test data is located in `/workspace/data/019/data/` directory, containing:
- `keys.txt` - Key matrices for testing
- `values.txt` - Value matrices for testing
- `queries.txt` - Query matrices for testing
- `ans.txt` - Expected answers

## Data Range

The provided data points and online data points have the **same shape**. Only the matrix internal data differs. For scoring, we will grade based on your program's performance, memory consumption, and your program's correctness.

### Performance Scoring Formula

If your program's correctness rate is `acc`, assuming your program takes time `t`, and your program consumes SRAM `m`, then your raw score will be:

$$1000000 \times acc \times \min(1.25, \frac{1.5 \times 10^{10}}{t}) \times e^{-\min(0.8,\frac{m}{1.5 \times 10^6})}$$

### Grade Composition

| Component | Percentage |
| :--: | :--: |
| OJ Score | 80% |
| Code Review | 20% |

The raw score from the formula above will be linearly mapped to 0-80 points, combined with code review (20 points) for a total of 100 points.

## Per-Testcase Resource Limits

- **Time Limit (per test case)**: 1000 ms
- **Memory Limit (per test case)**: 244 MiB
- **Disk Usage**: Disk access is not permitted

## Submission Requirements

### File Submission

For ACMOJ problem 2782, submit only the contents of `src.hpp` file.

### Submission Guidelines

- Submit only the `src.hpp` file containing your implementation
- Do not modify the provided function signature
- Ensure your implementation meets time and memory limits
- **You must use C++ language** to implement this assignment
- The local evaluation environment has gcc-13 and g++-13 available

### What to Implement

Your `src.hpp` file should contain:

```cpp
#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  // Your implementation here
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
```

You need to:
1. Implement the complete `Calculate` function
2. Use the GPU Simulator API correctly
3. Manage memory using the provided MatrixMemoryAllocator
4. Optimize for both execution time and SRAM usage

### Evaluation Notes

- The evaluation system will test your program's correctness, performance, and memory usage
- Score is calculated based on the formula provided in the Data Range section
- Optimize for both execution time and SRAM usage
- Please pay attention to the parallel execution of Calculation and IO instructions

### Academic Integrity

If any violations are found during evaluation or code review (including but not limited to using unconventional methods to pass test cases), your final score may be significantly reduced or become **0 points**.

