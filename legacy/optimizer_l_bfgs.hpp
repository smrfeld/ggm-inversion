//
/*
File: optimizer_l_bfgs.hpp
Created by: Oliver K. Ernst
Date: 12/4/20

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

#include "optimizer_base.hpp"

#ifndef OPTIMIZER_L_BFGS_H
#define OPTIMIZER_L_BFGS_H

namespace ggm {

class OptimizerLBFGS : public OptimizerBase {
private:
    
    bool _check_armijo_condition(double obj_func_0, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, double alpha, const arma::vec &updates, double c, double ax) const;
    bool _check_wolfe_condition(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, double alpha, const arma::vec &updates, double c, double ax) const;
    double _get_step_size_armijo_backtrack(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, const arma::vec &updates, const arma::vec &grads, double c=1e-4) const;
    double _get_step_size_wolfe_search(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, const arma::vec &updates, const arma::vec &grads, double c1=1e-4, double c2=0.9) const;

    struct RetSolveSGD {
        int opt_step_final;
        arma::vec s_vec, y_vec;
        arma::vec derivs_last, vals_last;
        arma::mat prec_mat_curr;
    };
    
    RetSolveSGD _solve_sgd_initial(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, int no_opt_steps, double lr_sgd_init) const;
    
public:
    
    int m = 5;
    double tol = 1e-10;
    double lr_sgd_init = 1e-10;
    int no_opt_steps = 100;
    Options options;
    
    using OptimizerBase::OptimizerBase;
    
    arma::mat solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const override;
};

}

#endif
