#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>

namespace pguresvt
{
    namespace utils
    {
        template <typename Arg, typename... Args>
        void print(std::ostream &out, Arg &&arg, Args &&... args)
        {
            out << std::forward<Arg>(arg);
            using expander = int[];
            (void)expander{0, (void(out << std::forward<Args>(args)), 0)...};
            out << std::endl;
        }

        template <typename Arg, typename... Args>
        void print_fixed(const int precision, Arg &&arg, Args &&... args)
        {
            print(std::cout, std::fixed, std::setprecision(precision), arg, args...);
        }
    } // namespace utils
} // namespace pguresvt
#endif