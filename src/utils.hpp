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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <cstdint>
#include <thread>
#include <vector>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pguresvt
{
    template <typename Arg, typename... Args>
    void Print(std::ostream &out, Arg &&arg, Args &&... args)
    {
        out << std::forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(out << std::forward<Args>(args)), 0)...};
        out << std::endl;
    }

    template <typename Arg, typename... Args>
    void PrintFixed(const uint32_t precision, Arg &&arg, Args &&... args)
    {
        Print(std::cout, std::fixed, std::setprecision(precision), arg, args...);
    }

    bool StrToBool(std::string &s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return (s.compare("1") == 0) || (s.compare("true") == 0);
    };

    void ParseParameters(std::istream &cfgfile, std::map<std::string, std::string> &options)
    {
        for (std::string line; std::getline(cfgfile, line);)
        {
            std::istringstream iss(line);
            std::string id, eq, val, temp;

            if (!(iss >> id))
            {
                continue; // Ignore empty lines
            }
            else if (id[0] == '#')
            {
                continue; // Ignore comment lines
            }
            else if (!(iss >> eq) || !(eq.compare(":")) || iss.get() != EOF)
            {
                while (iss >> temp)
                {
                    if (temp.find("#") != std::string::npos) // Support inline comments
                    {
                        break;
                    }
                    else
                    {
                        if (iss >> std::ws)
                        {
                            val += temp;
                        }
                        else
                        {
                            val += temp + " ";
                        }
                    }
                }
            }
            options[id] = val; // Set the parameter
        }
        return;
    }

    template <typename T>
    inline void SoftThreshold(arma::Col<T> &out, const arma::Col<T> &v, const arma::Col<T> &zeros, const double thresh)
    {
        out = arma::sign(v) % arma::max(arma::abs(v) - thresh, zeros);
    }

    template <typename T>
    inline void SoftThreshold(arma::Col<T> &out, const arma::Col<T> &v, const arma::Col<T> &zeros, const arma::Col<T> &thresh)
    {
        out = arma::sign(v) % arma::max(arma::abs(v) - thresh, zeros);
    }

    size_t sysrandom(void *dst, size_t dstlen)
    {
        // See StackOverflow - https://stackoverflow.com/a/45069417
        char *buffer = reinterpret_cast<char *>(dst);
        std::ifstream stream("/dev/urandom", std::ios_base::binary | std::ios_base::in);
        stream.read(buffer, dstlen);

        return dstlen;
    }

    template <typename Function, typename Integer_Type>
    void parallel(Function const &func,
                  Integer_Type dim_first,
                  Integer_Type dim_last,
                  int n_jobs = -1,
                  uint32_t threshold = 1)
    {
        if (n_jobs == 0) // No parallelization
        {
            for (auto a = dim_first; a != dim_last; ++a)
            {
                func(a);
            }
            return;
        }
        else // std::thread parallelization
        {
            uint32_t const total_cores = (n_jobs > 0) ? n_jobs : std::thread::hardware_concurrency();

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

} // namespace pguresvt

#endif