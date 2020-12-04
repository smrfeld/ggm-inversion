//
/*
File: optimizer.hpp
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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

namespace ggm {

class Optimizer {
private:
    
    std::vector<std::pair<int,int>> _idx_pairs_free;
    int _n_rows, _n_cols;
    
    /// Internal clean up
    void _clean_up();
    /// Internal copy
    void _copy(const Optimizer& other);
    /// Internal move
    void _move(Optimizer &other);

public:
    
    Optimizer(int n_rows, int n_cols, const std::vector<std::pair<int,int>> &idx_pairs_free);
    Optimizer(const Optimizer& other);
    Optimizer& operator=(const Optimizer& other);
    Optimizer(Optimizer&& other);
    Optimizer& operator=(Optimizer&& other);
    ~Optimizer();

    double get_obj_func_val(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;
    
    arma::mat get_derivs(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;
    
    arma::mat solve_sgd(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, double lr, int no_opt_steps) const;
    arma::mat solve_adam(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, double lr, int no_opt_steps, double adam_beta_1=0.9, double adam_beta_2=0.999, double adam_eps=1e-8) const;
};

}

#endif
