//
/*
File: helpers.hpp
Created by: Oliver K. Ernst
Date: 5/27/20

MIT License

Copyright (c) 2020 Oliver K. Ernst

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <string>
#include <armadillo>

#ifndef HELPERS_H
#define HELPERS_H

namespace ggm {

typedef arma::mat::fixed<3,2> VolBounds;
typedef arma::vec3 Pt;
typedef arma::uvec4 Tet;

/// Get random double
/// @details Stolen
///     https://stackoverflow.com/questions/2704521/generate-random-double-numbers-in-c
/// @param min Min
/// @param max Max
/// @return Random double
double get_random_number(double min, double max);

/// Get random int
/// @param min Min (inclusive)
/// @param max Max (inclusive)
/// @return Random int
int get_random_int(int min, int max);

/// Format a string
/// @details Stolen:
///     https://stackoverflow.com/questions/69738/c-how-to-get-fprintf-results-as-a-stdstring-w-o-sprintf#69911
/// @param fmt Format, followed by args
/// @return Formatted string
std::string format_str(const char *fmt, ...);

/// Format a string
/// @details Stolen:
///     https://stackoverflow.com/questions/69738/c-how-to-get-fprintf-results-as-a-stdstring-w-o-sprintf#69911
/// @param fmt Format, followed by args
/// @return Formatted string
std::string format_str(const std::string fmt, ...);

/// Format a string
/// @details Stolen:
///     https://stackoverflow.com/questions/69738/c-how-to-get-fprintf-results-as-a-stdstring-w-o-sprintf#69911
/// @param fmt Format, followed by args
/// @param ap Param list
/// @return Formatted string
std::string vformat(const char *fmt, va_list ap);

/// Clear entries in a file beyond an optimization step
/// @param fname Filename
/// @param opt_step Optimization step
void clear_entries_in_file_beyond_opt_step(std::string fname, int opt_step);

std::string _get_last_line_of_file(std::string fname);
std::string _get_line_of_file(std::string fname, int opt_step);
std::string get_line_of_file(std::string fname, int opt_step);

void write_mat(std::string fname, bool append, const arma::mat &mat);

void write_mat(std::string fname, int opt_step, bool append, const arma::mat &mat);

void write_mats(std::string fname, int opt_step, bool append, const arma::mat &mat1, const arma::mat &mat2);

void write_submat(std::string fname, bool append, const arma::mat &mat, const std::vector<std::pair<int,int>> &idx_pairs);

void write_submat(std::string fname, int opt_step, bool append, const arma::mat &mat, const std::vector<std::pair<int,int>> &idx_pairs);

void read_mat_from_line(std::string line, arma::mat &mat, int n_rows, int n_cols);
void read_mats_from_line(std::string line, arma::mat &mat1, int n_rows1, int n_cols1, arma::mat &mat2, int n_rows2, int n_cols2);

int _read_mat_from_vec(const std::vector<double> v, int idx_start, arma::mat &mat, int n_rows, int n_cols);

void _write_submat_to_stream(std::ofstream &f, const arma::mat &mat, const std::vector<std::pair<int,int>> &idx_pairs);
void _write_mat_to_stream(std::ofstream &f, const arma::mat &mat);

double get_min_eigenval(const arma::mat &mat);

void ensure_dir_exists(std::string dir);

};

#endif
