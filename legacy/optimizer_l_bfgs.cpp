//
/*
File: optimizer_base.cpp
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

#include "../include/ggm_inversion_bits/optimizer_l_bfgs.hpp"

namespace ggm {

bool OptimizerLBFGS::_check_armijo_condition(double obj_func_0, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, double alpha, const arma::vec &updates, double c, double ax) const {
    arma::mat cov_mat_step = arma::inv(prec_mat_curr + alpha * vec_to_mat(updates));
    return obj_func_0 - get_obj_func_val(cov_mat_step, cov_mat_true) >= alpha * c * ax;
}

bool OptimizerLBFGS::_check_wolfe_condition(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, double alpha, const arma::vec &updates, double c, double ax) const {
    arma::mat cov_mat_step = arma::inv(prec_mat_curr + alpha * vec_to_mat(updates));
    return - arma::dot(updates, get_deriv_vec(cov_mat_step, cov_mat_true)) <= - c * ax;
}

double OptimizerLBFGS::_get_step_size_armijo_backtrack(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, const arma::vec &updates, const arma::vec &grads, double c) const {
    
    double obj_func_0 = get_obj_func_val(arma::inv(prec_mat_curr), cov_mat_true);
    
    // This had better be a descent direction
    double ax = arma::dot(grads, updates);
    assert(ax < 0);
    
    double alpha = 1.0;
    while (!_check_armijo_condition(obj_func_0, prec_mat_curr, cov_mat_true, alpha, updates, c, ax)) {
        alpha *= 0.5;
    }
    
    return alpha;
}

double OptimizerLBFGS::_get_step_size_wolfe_search(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, const arma::vec &updates, const arma::vec &grads, double c1, double c2) const {
    
    double obj_func_0 = get_obj_func_val(arma::inv(prec_mat_curr), cov_mat_true);
    
    // This had better be a descent direction
    double ax = arma::dot(grads, updates);
    assert(ax < 0);
    
    double alpha = 100.0;
    while (
           (!_check_armijo_condition(obj_func_0, prec_mat_curr, cov_mat_true, alpha, updates, c1, ax))
           || (!_check_wolfe_condition(prec_mat_curr, cov_mat_true, alpha, updates, c2, ax))) {
        alpha *= 0.5;
        std::cout << alpha << std::endl;
    }
    
    return alpha;
}

OptimizerLBFGS::RetSolveSGD OptimizerLBFGS::_solve_sgd_initial(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, int no_opt_steps, double lr_sgd_init) const {
    
    arma::mat prec_mat_curr = prec_mat_init;
    arma::vec derivs_last, vals_last;

    for (int i=0; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
        
        arma::vec deriv_vec = get_deriv_vec(cov_mat_curr, cov_mat_true);
        
        // Check curvature
        // Except first step; first step is always SGD
        if (i != 0) {
            arma::vec s_vec = mat_to_vec(prec_mat_curr) - vals_last;
            arma::vec y_vec = deriv_vec - derivs_last;
            double curvature = arma::dot(s_vec, y_vec);
            std::cout << "*SGD* Curvature: " << curvature << std::endl;
            if (curvature > 0) {
                // Curvature OK; done with sgd
                RetSolveSGD ret;
                ret.opt_step_final = i;
                ret.s_vec = s_vec;
                ret.y_vec = y_vec;
                ret.derivs_last = derivs_last;
                ret.vals_last = vals_last;
                ret.prec_mat_curr = prec_mat_curr;
                
                return ret;
            }
        }
                
        // Advance
        vals_last = mat_to_vec(prec_mat_curr);
        derivs_last = deriv_vec;
        
        // Just SGD with small LR
        arma::vec update_vec = - lr_sgd_init * deriv_vec;
        prec_mat_curr += vec_to_mat(update_vec);
    }
    
    // Fail
    throw std::runtime_error("Could not find positive curvature direction after max no of optimization steps of SGD");
}

arma::mat OptimizerLBFGS::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const {

    // Solve sgd first
    auto ret_sgd = _solve_sgd_initial(cov_mat_true, prec_mat_init, no_opt_steps, lr_sgd_init);
    
    // Start opt step here
    int opt_step_start = ret_sgd.opt_step_final;
    std::cout << "SGD gave positive curvature direction after: " << opt_step_start << " steps." << std::endl;
    
    // Prec mat
    arma::mat prec_mat_curr = ret_sgd.prec_mat_curr;
    
    // S, y
    // k-1 : (val at k) - (val at k-1)
    std::vector<arma::vec> s_st, y_st;
    
    // Derivs and vals last
    arma::vec derivs_last = ret_sgd.derivs_last;
    arma::vec vals_last = ret_sgd.vals_last;
    
    // Init prev obj func to check convergence
    double obj_func_prev = 1e16;
    
    for (int i=opt_step_start; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);

        // Log progress if needed
        _log_progress_if_needed(options, i, no_opt_steps, cov_mat_curr, cov_mat_true, prec_mat_curr);
        
        // Write progress if needed
        _write_progress_if_needed(options, i, prec_mat_curr, cov_mat_curr, cov_mat_true);
        
        // Check convergence
        double obj_func_new = get_obj_func_val(cov_mat_curr, cov_mat_true);
        if (abs(obj_func_new - obj_func_prev) < tol) {
            std::cout << "*LBFGS* Converged: obj func val: " << obj_func_new << " prev: " << obj_func_prev << " change: " << abs(obj_func_new - obj_func_prev) << " < " << tol << std::endl;
            break;
        }
        
        // Calculate derivs
        arma::vec deriv_vec = get_deriv_vec(cov_mat_curr, cov_mat_true);
        
        // Set change
        // k-1 : (val at k) - (val at k-1)
        arma::vec s_vec = mat_to_vec(prec_mat_curr) - vals_last;
        arma::vec y_vec = deriv_vec - derivs_last;
        double curvature = arma::dot(s_vec, y_vec);
        std::cout << "*LBFGS* Curvature: " << curvature << std::endl;
        if (curvature > 0) {
            s_st.push_back(s_vec);
            y_st.push_back(y_vec);
            /*
            s_map[i-1] = s_vec;
            y_map[i-1] = y_vec;
             */
        } else {
            // Skip!
            std::cout << "*LBFGS* Curvature condition not satisfied; not including pair" << std::endl;
        }
                    
        // Get update
        int idx_start = s_st.size()-1;
        int idx_end = 0;
        // int idx_start = std::max(i-1,first_entry);
        // int idx_end = std::max(i-m,first_entry);
        std::map <int, double> rho;
        std::map <int, double> alpha;
        
        // self.log.debug("   Loop #1 range: %s" % list(reversed(range(idx_end, idx_start))))
        
        double eps=1e-12;
        
        arma::vec q = deriv_vec;
        for (int i=idx_start; i>=idx_end; i--) {
            rho[i] = 1.0 / ( arma::as_scalar(arma::dot(y_st.at(i), s_st.at(i))) + eps);
            alpha[i] = rho.at(i) * arma::dot(s_st.at(i), q);
            q = q - alpha.at(i) * y_st.at(i);
        }
        
        arma::vec s_back = s_st.back();
        arma::vec y_back = y_st.back();
        double gamma_k = arma::as_scalar(arma::dot(s_back, y_back)) / ( arma::as_scalar(arma::dot(y_back, y_back)) + eps);
        
        arma::vec z = gamma_k * q;

        idx_start = 0;
        idx_end = s_st.size();
        // idx_start = std::max(i-m,first_entry);
        // idx_end = i;
        // self.log.debug("   Loop #2 range: %s" % list(range(idx_start, idx_end)))
        for (int i=idx_start; i<idx_end; i++) {
            double beta_i = rho.at(i) * arma::as_scalar(arma::dot(y_st.at(i),z));
            z = z + s_st.at(i) * (alpha.at(i) - beta_i);
        }
        
        // Update vec
        arma::vec update_vec = -z;
        
        // Remove old
        if (s_st.size() > m) {
            s_st.erase(s_st.begin());
            y_st.erase(y_st.begin());
        }
        /*
        int idx_remove = i - m - 1;
        if (idx_remove >= first_entry) {
            s_map.erase(idx_remove);
            y_map.erase(idx_remove);
        }
         */
        
        // Advance
        vals_last = mat_to_vec(prec_mat_curr);
        derivs_last = deriv_vec;
        
        // Update
        double step_size = _get_step_size_armijo_backtrack(prec_mat_curr, cov_mat_true, update_vec, deriv_vec);
        // double step_size = _get_step_size_wolfe_search(prec_mat_curr, cov_mat_true, update_vec, deriv_vec);
        
        std::cout << "Opt step: " << i << " / " << no_opt_steps << " - step size: " << step_size << " - Obj func: " << obj_func_new << std::endl;
        
        prec_mat_curr += step_size * vec_to_mat(update_vec);
        
        // Advance obj func
        obj_func_prev = obj_func_new;
    }
    
    return prec_mat_curr;
}

};


