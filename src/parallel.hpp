/***************************************************************************

  Copyright (C) 2015-2020 Tom Furnival

  This file is part of  PGURE-SVT.

  PGURE-SVT is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  PGURE-SVT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PGURE-SVT. If not, see <http://www.gnu.org/licenses/>.

***************************************************************************/

#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include <cstdint>
#include <thread>
#include <vector>

template <typename Function, typename Integer_Type>
void parallel(Function const &func,
              Integer_Type dim_first,
              Integer_Type dim_last,
              uint32_t threshold = 1,
              bool parallel_mode = true)
{
  if (!parallel_mode)
  {
    for (auto a = dim_first; a != dim_last; ++a)
    {
      func(a);
    }
    return;
  }
  else
  {
    uint32_t const total_cores = std::thread::hardware_concurrency();

    if ((total_cores <= 1) || ((dim_last - dim_first) <= threshold)) // case of non-parallel or small jobs
    {
      for (auto a = dim_first; a != dim_last; ++a)
      {
        func(a);
      }
      return;
    }

    std::vector<std::thread> threads;
    if (dim_last - dim_first <= total_cores) // case of small job numbers
    {
      for (auto index = dim_first; index != dim_last; ++index)
        threads.emplace_back(std::thread{[&func, index]() { func(index); }});
      for (auto &th : threads)
      {
        th.join();
      }
      return;
    }

    auto const &job_slice = [&func](Integer_Type a, Integer_Type b) { // case of more jobs than CPU cores
      if (a >= b)
      {
        return;
      }
      while (a != b)
      {
        func(a++);
      }
    };

    threads.reserve(total_cores - 1);
    uint64_t tasks_per_thread = (dim_last - dim_first + total_cores - 1) / total_cores;

    for (auto index = 0UL; index != total_cores - 1; ++index)
    {
      Integer_Type first = tasks_per_thread * index + dim_first;
      first = std::min(first, dim_last);
      Integer_Type last = first + tasks_per_thread;
      last = std::min(last, dim_last);
      threads.emplace_back(std::thread{job_slice, first, last});
    }

    job_slice(tasks_per_thread * (total_cores - 1), dim_last);
    for (auto &th : threads)
    {
      th.join();
    }
  }
}

template <typename Function, typename Integer_Type>
void parallel(Function const &func, Integer_Type dim_last)
{
  parallel(func, Integer_Type{0}, dim_last);
}

#endif

/*
TODO: support OMP

include <omp.h>

// Set up OMP
#if defined(_OPENMP)
  omp_set_dynamic(0);
  omp_set_num_threads(numthreads);
#endif

#pragma omp parallel for shared(v, weights) \
            private(block, Ublock, Sblock, Vblock)
*/
