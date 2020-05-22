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

namespace pguresvt
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
    void printFixed(const uint32_t precision, Arg &&arg, Args &&... args)
    {
        print(std::cout, std::fixed, std::setprecision(precision), arg, args...);
    }

    bool strToBool(std::string &s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return (s.compare("1") == 0) || (s.compare("true") == 0);
    };

    inline double median(const arma::mat &X)
    {
        return arma::median(arma::median(X));
    }

    inline double mean(const arma::mat &X)
    {
        return arma::mean(arma::mean(X));
    }

    inline void softThreshold(arma::vec &out, const arma::vec &v, const arma::vec &zeros, const double thresh)
    {
        out = arma::sign(v) % arma::max(arma::abs(v) - thresh, zeros);
    }

    inline void softThreshold(arma::vec &out, const arma::vec &v, const arma::vec &zeros, const arma::vec &thresh)
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

} // namespace pguresvt

#endif