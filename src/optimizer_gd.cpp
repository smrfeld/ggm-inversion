//
/*
File: optimizer_l_bfgs.cpp
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

#include "../include/ggm_inversion_bits/optimizer_gd.hpp"

namespace ggm {

arma::mat OptimizerGD::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const {

    arma::mat prec_mat_curr = prec_mat_init;
        
    for (size_t i=0; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
                    
        // Log if needed
        _log_progress_if_needed(options, i, no_opt_steps, cov_mat_curr, cov_mat_true, prec_mat_curr);
        
        // Write if needed
        _write_progress_if_needed(options, i, prec_mat_curr, cov_mat_curr, cov_mat_true);
        
        arma::mat derivs = get_deriv_mat(cov_mat_curr, cov_mat_true);
        prec_mat_curr -= lr * derivs;
    }
    
    return prec_mat_curr;
}

}
