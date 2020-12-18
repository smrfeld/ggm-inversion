//
/*
File: optimizer_base.hpp
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

#ifndef OPTIMIZER_BASE_H
#define OPTIMIZER_BASE_H

namespace ginv {

class L2OptimizerBase {
        
protected:
    
    std::vector<std::pair<int,int>> _idx_pairs_free;
    int _dim;
    
    double _get_first_deriv_inverse_mat(const arma::mat &cov_mat_curr, int d1, int d2, int n1, int n2) const;
    double _get_second_deriv_inverse_mat(const arma::mat &cov_mat_curr, int d1, int d2, int d3, int d4, int n1, int n2) const;

    void _log_progress_if_needed(Options options, int opt_step, int no_opt_steps, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_targets, const arma::mat &prec_mat_curr) const;
    
    void _write_progress_if_needed(Options options, int opt_step, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;
    
private:
    
    /// Internal clean up
    void _clean_up();
    /// Internal copy
    void _copy(const L2OptimizerBase& other);
    /// Internal move
    void _move(L2OptimizerBase &other);

public:
    
    L2OptimizerBase(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free);
    L2OptimizerBase(const L2OptimizerBase& other);
    L2OptimizerBase& operator=(const L2OptimizerBase& other);
    L2OptimizerBase(L2OptimizerBase&& other);
    L2OptimizerBase& operator=(L2OptimizerBase&& other);
    virtual ~L2OptimizerBase();

    arma::mat vec_to_mat(const arma::vec &vec) const;
    arma::vec mat_to_vec(const arma::mat &mat) const;
    
    double get_obj_func_val(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;
    
    arma::mat get_deriv_mat(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;
    arma::vec get_deriv_vec(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;

    arma::mat get_hessian(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;
    
    virtual arma::mat solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const = 0;
};

}

#endif
