//
/*
File: optimizer_adam.hpp
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

#include "../include/ggm_inversion_bits/l2_optimizer_adam.hpp"

namespace ginv {
        
arma::mat L2OptimizerAdam::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const {
    
    arma::mat prec_mat_curr = prec_mat_init;

    arma::mat adam_mt, adam_vt;

    for (size_t i=0; i<no_opt_steps; i++) {
                
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
        
        // Log
        _log_progress_if_needed(options, i, no_opt_steps, cov_mat_curr, cov_mat_true, prec_mat_curr);
        
        // Write
        _write_progress_if_needed(options, i, prec_mat_curr, cov_mat_curr, cov_mat_true);
        
        arma::mat derivs = get_deriv_mat(cov_mat_curr, cov_mat_true);

        if (i == 0) {
            adam_mt = derivs;
        } else {
            adam_mt = adam_beta_1 * adam_mt + (1 - adam_beta_1) * derivs;
        }

        if (i == 0) {
            adam_vt = pow(derivs,2);
        } else {
            adam_vt = adam_beta_2 * adam_vt + (1 - adam_beta_2) * pow(derivs,2);
        }
                                
        arma::mat adam_mt_corr = adam_mt / (1 - pow(adam_beta_1, i+1));
        arma::mat adam_vt_corr = adam_vt / (1 - pow(adam_beta_2, i+1));

        prec_mat_curr -= lr * adam_mt_corr / (sqrt(adam_vt_corr) + adam_eps);
    }

    return prec_mat_curr;
}

}
