//
/*
File: helpers.cpp
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

#include "../include/ggm_inversion_bits/helpers.hpp"

#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include <sstream>
#include <ostream>
#include <istream>
#include <fstream>
#include <random>

namespace ggm {

double get_random_number(double min, double max)
{
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

int get_random_int(int min, int max) {
    if (min == max) {
        return min;
    }
    
    // Eg min = 13
    // Max = 23
    // Width = 11
    int width = max - min + 1;
    // Between 0 and 10
    int i = rand() % width;
    // Between 13 and 23
    i += min;
    return i;
}

std::string format_str(const std::string fmt, ...) {
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat (fmt.c_str(), ap);
    va_end (ap);
    return buf;
}

std::string format_str(const char *fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat (fmt, ap);
    va_end (ap);
    return buf;
}

std::string vformat(const char *fmt, va_list ap)
{
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.
    size_t size = 1024;
    char buf[size];

    // Try to vsnprintf into our buffer.
    va_list apcopy;
    va_copy (apcopy, ap);
    int needed = vsnprintf (&buf[0], size, fmt, ap);
    // NB. On Windows, vsnprintf returns -1 if the string didn't fit the
    // buffer.  On Linux & OSX, it returns the length it would have needed.

    if (needed <= size && needed >= 0) {
        // It fit fine the first time, we're done.
        return std::string (&buf[0]);
    } else {
        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So do a malloc of the right size and try again.
        // This doesn't happen very often if we chose our initial size
        // well.
        std::vector <char> buf;
        size = needed;
        buf.resize (size);
        needed = vsnprintf (&buf[0], size, fmt, apcopy);
        return std::string (&buf[0]);
    }
}

void clear_entries_in_file_beyond_opt_step(std::string fname, int opt_step) {
    
    std::ifstream fin;
    fin.open(fname, std::ifstream::in);

    if (!fin.is_open()) {
        return;
    }
    
    std::istringstream iss;
    std::string line;
    std::vector<std::string> lines;
    std::string opt_step_read_str;
    int opt_step_read;

    bool finished = false;
    while ((!finished) && (getline(fin,line))) {
        
        if (line == "") { continue; };
        iss = std::istringstream(line);
        iss >> opt_step_read_str;

        // Convert
        opt_step_read = atoi(opt_step_read_str.c_str());
        
        // Check
        if (opt_step_read > opt_step) {
            // Stop
            finished = true;
        } else {
            // Append
            lines.push_back(line);
        }
                
        // Reset
        opt_step_read_str="";
    }
    
    fin.close();

    // Write back
    std::ofstream fout;
    fout.open(fname, std::ofstream::out);
    
    if (!fout.is_open()) {
        throw std::invalid_argument("File: " + fname + " does not exist for writing.");
    }

    // Write
    for (auto l: lines) {
        fout << l << "\n";
    }
    
    fout.close();

}

std::string _get_last_line_of_file(std::string fname) {
    std::ifstream fin;
    fin.open(fname, std::ifstream::in);
    
    if (!fin.is_open()) {
        throw std::invalid_argument("File: " + fname + " does not exist for reading.");
    }
    
    fin.seekg(0,std::ios_base::end);      //Start at end of file
    char ch = ' ';                        //Init ch not equal to '\n'
    while(ch != '\n'){
        fin.seekg(-2,std::ios_base::cur); //Two steps back, this means we
        //will NOT check the last character
        if((int)fin.tellg() <= 0){        //If passed the start of the file,
            fin.seekg(0);                 //this is the start of the line
            break;
        }
        fin.get(ch);                      //Check the next character
    }
    
    std::string lastLine;
    getline(fin,lastLine);                      // Read the current line
    
    fin.close();
    
    return lastLine;
}

std::string _get_line_of_file(std::string fname, int opt_step) {
        
    std::ifstream fin;
    fin.open(fname, std::ifstream::in);
    
    if (!fin.is_open()) {
        throw std::invalid_argument("File: " + fname + " does not exist for reading.");
    }
    
    std::istringstream iss;
    std::string line;
    std::string opt_step_read_str;
    int opt_step_read;
    
    while (getline(fin,line)) {
        
        if (line == "") { continue; };
        iss = std::istringstream(line);
        iss >> opt_step_read_str;
        
        // Convert
        opt_step_read = atoi(opt_step_read_str.c_str());
        
        // Check
        if (opt_step == opt_step_read) {
            fin.close();
            return line;
        }
        
        // Reset
        opt_step_read_str="";
    }
    
    // Fail if we get here
    fin.close(); // close!
    throw std::invalid_argument("Could not find opt step in param file.");
}

std::string get_line_of_file(std::string fname, int opt_step) {
    if (opt_step == -1) {
        return _get_last_line_of_file(fname);
    } else {
        return _get_line_of_file(fname, opt_step);
    }
}

void _write_mat_to_stream(std::ofstream &f, const arma::mat &mat) {
    f << std::setprecision(16);
    for (auto i=0; i<mat.n_rows; i++) {
        for (auto j=0; j<mat.n_cols; j++) {
            f << " " << mat(i,j);
        }
    }
}

void write_mat(std::string fname, bool append, const arma::mat &mat) {
    std::ofstream f;
    if (append) {
        f.open(fname, std::ofstream::app);
    } else {
        f.open(fname, std::ofstream::out);
    }
    
    if (!f.is_open()) {
        throw std::invalid_argument("File: " + fname + " does not exist for writing.");
    }
    
    // Opt step and rate
    _write_mat_to_stream(f, mat);
    
    // newline
    f << "\n";
    
    f.close();
}

void write_mat(std::string fname, int opt_step, bool append, const arma::mat &mat) {
    std::ofstream f;
    if (append) {
        f.open(fname, std::ofstream::app);
    } else {
        f.open(fname, std::ofstream::out);
    }
    
    if (!f.is_open()) {
        throw std::invalid_argument("File: " + fname + " does not exist for writing.");
    }
    
    // Opt step and rate
    f << opt_step;
    _write_mat_to_stream(f, mat);
    
    // newline
    f << "\n";
    
    f.close();
}

void write_mats(std::string fname, int opt_step, bool append, const arma::mat &mat1, const arma::mat &mat2) {
    
    std::ofstream f;
    if (append) {
        f.open(fname, std::ofstream::app);
    } else {
        f.open(fname, std::ofstream::out);
    }
    
    if (!f.is_open()) {
        throw std::invalid_argument("File: " + fname + " does not exist for writing.");
    }
    
    // Opt step and rate
    f << opt_step;
    _write_mat_to_stream(f, mat1);
    _write_mat_to_stream(f, mat2);

    // newline
    f << "\n";
    
    f.close();
}

int _read_mat_from_vec(const std::vector<double> v, int idx_start, arma::mat &mat, int n_rows, int n_cols) {
    mat = arma::mat(n_rows, n_cols);
    int idx = idx_start;
    for (auto i=0; i<n_rows; i++) {
        for (auto j=0; j<n_cols; j++) {
            mat(i,j) = v[idx];
            idx += 1;
        }
    }
    
    return idx;
}

void read_mat_from_line(std::string line, arma::mat &mat, int n_rows, int n_cols) {
    
    // If possible, always prefer std::vector to naked array
    std::vector<double> v;
    
    // Build an istream that holds the input string
    std::istringstream iss(line);
    
    // Iterate over the istream, using >> to grab floats
    // and push_back to store them in the vector
    std::copy(std::istream_iterator<double>(iss),
              std::istream_iterator<double>(),
              std::back_inserter(v));
    
    int idx = 1;
    idx = _read_mat_from_vec(v, idx, mat, n_rows, n_cols);
}

void read_mats_from_line(std::string line, arma::mat &mat1, int n_rows1, int n_cols1, arma::mat &mat2, int n_rows2, int n_cols2) {
    
    // If possible, always prefer std::vector to naked array
    std::vector<double> v;
    
    // Build an istream that holds the input string
    std::istringstream iss(line);
    
    // Iterate over the istream, using >> to grab floats
    // and push_back to store them in the vector
    std::copy(std::istream_iterator<double>(iss),
              std::istream_iterator<double>(),
              std::back_inserter(v));
    
    int idx = 1;
    idx = _read_mat_from_vec(v, idx, mat1, n_rows1, n_cols1);
    idx = _read_mat_from_vec(v, idx, mat2, n_rows2, n_cols2);
}

double get_min_eigenval(const arma::mat &mat) {
    auto eigen = arma::eig_gen(mat);
    
    arma::vec eigen_real(eigen.n_rows);
    for (auto i=0; i<eigen.n_rows; i++) {
        eigen_real(i) = eigen(i).real();
    }
    double min_eigenval = arma::min(eigen_real);
    
    return min_eigenval;
}

};


