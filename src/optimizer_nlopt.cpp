//
/*
File: optimizer_nlopt.cpp
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

#include "../include/ggm_inversion_bits/optimizer_nlopt.hpp"

namespace ggm {

double nlopt_obj_func(const std::vector<double> &prec_mat_std_vec, std::vector<double> &deriv_std_vec, void* input_obj_func_val) {
    
    // Get input
    InputObjFuncVal* input = static_cast<InputObjFuncVal*>(input_obj_func_val);
    const OptimizerNLOpt *optimizer = input->optimizer_nlopt;
    arma::mat cov_mat_true = input->cov_mat_true;
    
    arma::mat prec_mat_curr = optimizer->std_vec_to_mat(prec_mat_std_vec);
    arma::mat cov_mat_curr = arma::inv_sympd(prec_mat_curr);
    
    // Obj func val
    double obj_func_val = optimizer->get_obj_func_val(cov_mat_curr, cov_mat_true);
    
    // Get deriv
    arma::mat deriv_mat = optimizer->get_deriv_mat(cov_mat_curr, cov_mat_true);
    deriv_std_vec = optimizer->mat_to_std_vec(deriv_mat);
    
    return obj_func_val;
}

arma::mat OptimizerNLOpt::std_vec_to_mat(const std::vector<double> &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        mat(pr.first, pr.second) = vec.at(i);
        mat(pr.second, pr.first) = vec.at(i);
    }
    
    return mat;
}

std::vector<double> OptimizerNLOpt::mat_to_std_vec(const arma::mat &mat) const {
    
    std::vector<double> vec(_idx_pairs_free.size());
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        vec[i] = mat(pr.first, pr.second);
    }
    
    return vec;
}

arma::mat OptimizerNLOpt::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, int no_opt_steps, LogOptions log_options, WritingOptions writing_options) const {

    int no_params = _idx_pairs_free.size();
    nlopt::opt opt(algorithm, no_params);
    
    // Set obj func
    InputObjFuncVal *input = new InputObjFuncVal();
    input->optimizer_nlopt = this;
    input->cov_mat_true = cov_mat_true;
    opt.set_min_objective(&nlopt_obj_func, input);

    // Tolerance
    opt.set_ftol_abs(tol);
    
    // Run
    std::vector<double> prec_mat_init_std_vec = mat_to_std_vec(prec_mat_init);
    double obj_func_final;
    auto res = opt.optimize(prec_mat_init_std_vec, obj_func_final);
    
    // Clean up!
    delete input;
}

};


