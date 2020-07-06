/**********************
# Author: Tom Furnival
# License: GPLv3
***********************/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <chrono>
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

    double ElapsedSeconds(std::chrono::high_resolution_clock::time_point t0,
                          std::chrono::high_resolution_clock::time_point t1)
    {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 1E-6);
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

    template <typename Function, typename Integer_Type>
    void parallel(Function const &func,
                  Integer_Type dimFirst,
                  Integer_Type dimLast,
                  int nJobs = -1,
                  uint32_t threshold = 1)
    {
        uint32_t const totalCores = (nJobs > 0) ? nJobs : std::thread::hardware_concurrency();

        if ((nJobs == 0) || (totalCores <= 1) || ((dimLast - dimFirst) <= threshold)) // No parallelization or small jobs
        {
            for (auto a = dimFirst; a != dimLast; ++a)
            {
                func(a);
            }
            return;
        }
        else // std::thread parallelization
        {
            std::vector<std::thread> threads;
            if (dimLast - dimFirst <= totalCores) // case of small job numbers
            {
                for (auto index = dimFirst; index != dimLast; ++index)
                    threads.emplace_back(std::thread{[&func, index]() { func(index); }});
                for (auto &th : threads)
                {
                    th.join();
                }
                return;
            }

            auto const &jobSlice = [&func](Integer_Type a, Integer_Type b) { // case of more jobs than CPU cores
                if (a >= b)
                {
                    return;
                }
                while (a != b)
                {
                    func(a++);
                }
            };

            threads.reserve(totalCores - 1);
            uint64_t tasksPerThread = (dimLast - dimFirst + totalCores - 1) / totalCores;

            for (auto index = 0UL; index != totalCores - 1; ++index)
            {
                Integer_Type first = tasksPerThread * index + dimFirst;
                first = std::min(first, dimLast);
                Integer_Type last = first + tasksPerThread;
                last = std::min(last, dimLast);
                threads.emplace_back(std::thread{jobSlice, first, last});
            }

            jobSlice(tasksPerThread * (totalCores - 1), dimLast);
            for (auto &th : threads)
            {
                th.join();
            }
        }
    };

} // namespace pguresvt

template <typename T>
void SetMemStateMat(T &t, int state)
{
    const_cast<arma::uhword &>(t.mem_state) = state;
}

template <typename T>
void SetMemStateCube(T &t, int state)
{
    const_cast<arma::uword &>(t.mem_state) = state;
}

template <typename T>
size_t GetMemState(T &t)
{
    if ((t.mem) && (t.n_elem <= arma::arma_config::mat_prealloc))
    {
        return 0;
    }
    return static_cast<size_t>(t.mem_state);
}

template <typename T>
inline typename T::elem_type *GetMemory(T &m)
{
    if ((m.mem) && (m.n_elem <= arma::arma_config::mat_prealloc))
    {
        typename T::elem_type *mem = arma::memory::acquire<typename T::elem_type>(m.n_elem);
        arma::arrayops::copy(mem, m.memptr(), m.n_elem);
        return mem;
    }
    else
    {
        return m.memptr();
    }
}

#endif