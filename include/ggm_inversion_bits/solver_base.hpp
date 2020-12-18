//
/*
File: solver_base.hpp
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

#include "options.hpp"

#include <string>
#include <armadillo>

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

namespace ginv {

class SolverBase {
        
protected:
    
    std::vector<std::pair<int,int>> _idx_pairs_free, _idx_pairs_non_free;
    int _dim;
    
private:
    
    bool _check_pair_exists(const std::vector<std::pair<int,int>> &pairs, std::pair<int,int> pr_search) const;
    
    /// Internal clean up
    void _clean_up();
    /// Internal copy
    void _copy(const SolverBase& other);
    /// Internal move
    void _move(SolverBase &other);

public:
    
    SolverBase(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free);
    SolverBase(const SolverBase& other);
    SolverBase& operator=(const SolverBase& other);
    SolverBase(SolverBase&& other);
    SolverBase& operator=(SolverBase&& other);
    virtual ~SolverBase();
    
    std::vector<std::pair<int,int>> get_idx_pairs_free() const;
    int get_dim() const;
    
    arma::mat free_vec_to_mat(const arma::vec &vec) const;
    arma::mat non_free_vec_to_mat(const arma::vec &vec) const;
    arma::vec free_mat_to_vec(const arma::mat &mat) const;
    arma::vec non_free_mat_to_vec(const arma::mat &mat) const;

    arma::mat zero_free_elements(const arma::mat &mat) const;
    arma::mat zero_non_free_elements(const arma::mat &mat) const;

    virtual std::pair<arma::mat,arma::mat> solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const = 0;
};

}

#endif
