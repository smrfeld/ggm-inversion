//
/*
File: optimizer_optim.cpp
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

#include "../include/ggm_inversion_bits/optimizer_optim.hpp"

namespace ggm {

double optim_obj_func(const arma::vec &prec_mat_vec, arma::vec *deriv_vec, void* input_obj_func_val) {

    // Get input
    InputObjFuncVal* input = static_cast<InputObjFuncVal*>(input_obj_func_val);
    const OptimizerOptim *optimizer = input->optimizer;
    arma::mat cov_mat_true = input->cov_mat_true;
    
    arma::mat prec_mat_curr = optimizer->vec_to_mat(prec_mat_vec);
    arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
    
    // Obj func val
    double obj_func_val = optimizer->get_obj_func_val(cov_mat_curr, cov_mat_true);
    
    // Get deriv
    if (deriv_vec != nullptr) {
        arma::mat deriv_mat = optimizer->get_deriv_mat(cov_mat_curr, cov_mat_true);
        *deriv_vec = optimizer->mat_to_vec(deriv_mat);
    }
    
    return obj_func_val;
}

arma::mat OptimizerOptim::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const {
    
    // Init
    arma::vec prec_mat_vec = mat_to_vec(prec_mat_init);
    
    // Optional input
    InputObjFuncVal *input = new InputObjFuncVal();
    input->optimizer = this;
    input->cov_mat_true = cov_mat_true;
        
    // Solve
    bool success;
    if (_alg == OptimAlg::lbfgs) {
        // LBFS
        success = optim::lbfgs(prec_mat_vec, optim_obj_func, input, settings);
    } else if (_alg == OptimAlg::bfgs) {
        // BFGS
        success = optim::bfgs(prec_mat_vec, optim_obj_func, input, settings);
    } else if (_alg == OptimAlg::cg) {
        // CG
        success = optim::cg(prec_mat_vec, optim_obj_func, input, settings);
    } else {
        // GD
        success = optim::gd(prec_mat_vec, optim_obj_func, input, settings);
    }
    
    // Must succeed
    if (log_result) {
        std::cout << "--- Result ---" << std::endl;
        if (!success) {
            std::cerr << "Failed to converge after: " << settings.opt_iter << " iterations" << std::endl;
        } else {
            std::cout << "Converged after: " << settings.opt_iter << " iterations" << std::endl;
        }
        std::cout << "Obj func value: " << settings.opt_fn_value << std::endl;
        std::cout << "Error value: " << settings.opt_error_value << std::endl;
        std::cout << "Prec mat: " << std::endl;
        std::cout << vec_to_mat(prec_mat_vec) << std::endl;
        std::cout << "Cov mat: " << std::endl;
        std::cout << arma::inv(vec_to_mat(prec_mat_vec)) << std::endl;
    }
    
    // Clean up!
    delete input;
    
    return vec_to_mat(prec_mat_vec);
}

void OptimizerOptim::set_alg_adam(double lr) {
    _alg = OptimAlg::adam;
    settings.gd_settings.method = 6;
    settings.gd_settings.par_step_size = lr;
}

void OptimizerOptim::set_alg_lbfgs() {
    _alg = OptimAlg::lbfgs;
}

void OptimizerOptim::set_alg_bfgs() {
    _alg = OptimAlg::bfgs;
}

void OptimizerOptim::set_alg_sgd(double lr){
    _alg = OptimAlg::sgd;
    settings.gd_settings.method = 0;
    settings.gd_settings.par_step_size = lr;
}

void OptimizerOptim::set_alg_cg(double lr){
    _alg = OptimAlg::cg;
    settings.gd_settings.par_step_size = lr;
}

};


