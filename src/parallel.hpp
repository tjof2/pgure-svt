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

#include <thread>
#include <vector>

constexpr unsigned long parallel_mode = 1;

template <typename Function, typename Integer_Type>
void parallel(Function const &func, Integer_Type dim_first,
              Integer_Type dim_last, unsigned long threshold = 1)
{
  if constexpr (parallel_mode == 0)
  {
    for (auto a : range(dim_first, dim_last))
      func(a);
    return;
  }
  else // for constexpr-if, must have else
  {
    unsigned int const total_cores = std::thread::hardware_concurrency();

    // case of non-parallel or small jobs
    if ((total_cores <= 1) || ((dim_last - dim_first) <= threshold))
    {
      for (auto a = dim_first; a != dim_last; ++a)
        func(a);
      return;
    }

    // case of small job numbers
    std::vector<std::thread> threads;
    if (dim_last - dim_first <= total_cores)
    {
      for (auto index = dim_first; index != dim_last; ++index)
        threads.emplace_back(std::thread{[&func, index]() { func(index); }});
      for (auto &th : threads)
        th.join();
      return;
    }

    // case of more jobs than CPU cores
    auto const &job_slice = [&func](Integer_Type a, Integer_Type b) {
      if (a >= b)
        return;
      while (a != b)
        func(a++);
    };

    threads.reserve(total_cores - 1);
    std::uint_least64_t tasks_per_thread =
        (dim_last - dim_first + total_cores - 1) / total_cores;

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
      th.join();
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
