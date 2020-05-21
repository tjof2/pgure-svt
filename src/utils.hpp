#ifndef UTILS_HPP
#define UTILS_HPP

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
    void print_fixed(const uint32_t precision, Arg &&arg, Args &&... args)
    {
        print(std::cout, std::fixed, std::setprecision(precision), arg, args...);
    }

    bool strToBool(std::string const &s)
    {
        // Little function to convert string "0"/"1" to boolean
        return s != "0";
    };

    inline double median(const arma::mat &X)
    {
        return arma::median(arma::median(X));
    }

    inline double mean(const arma::mat &X)
    {
        return arma::mean(arma::mean(X));
    }

    inline void SoftThreshold(arma::vec &out, const arma::vec &v, const arma::vec &zeros, const double thresh)
    {
        out = arma::sign(v) % arma::max(arma::abs(v) - thresh, zeros);
    }

    inline void SoftThreshold(arma::vec &out, const arma::vec &v, const arma::vec &zeros, const arma::vec &thresh)
    {
        out = arma::sign(v) % arma::max(arma::abs(v) - thresh, zeros);
    }

} // namespace pguresvt

#endif