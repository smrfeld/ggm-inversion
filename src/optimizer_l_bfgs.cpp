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
    arma::mat cov_mat_step = arma::inv(prec_mat_curr + alpha * _vec_to_mat(updates));
    return obj_func_0 - get_obj_func_val(cov_mat_step, cov_mat_true) >= alpha * c * ax;
}

bool OptimizerLBFGS::_check_wolfe_condition(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, double alpha, const arma::vec &updates, double c, double ax) const {
    arma::mat cov_mat_step = arma::inv(prec_mat_curr + alpha * _vec_to_mat(updates));
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

arma::mat OptimizerLBFGS::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, int no_opt_steps, int m, double tol) const {
 
    arma::mat prec_mat_curr = prec_mat_init;
    std::map<int,arma::vec> s_map, y_map;
    arma::vec derivs_last, vals_last;
    double obj_func_prev = 1e16;
    
    for (int i=0; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);

        // Check convergence
        double obj_func_new = get_obj_func_val(cov_mat_curr, cov_mat_true);
        if (abs(obj_func_new - obj_func_prev) < tol) {
            std::cout << "Converged: obj func val: " << obj_func_new << " prev: " << obj_func_prev << " change: " << abs(obj_func_new - obj_func_prev) << " < " << tol << std::endl;
            break;
        }
        
        arma::vec deriv_vec = get_deriv_vec(cov_mat_curr, cov_mat_true);
        
        arma::vec update_vec;
        if (i == 0) {
            // Just SGD with small LR
            update_vec = - 1e-10 * deriv_vec;
        } else {
            
            // Set change
            // k-1 : (val at k) - (val at k-1)
            arma::vec s_vec = _mat_to_vec(prec_mat_curr) - vals_last;
            arma::vec y_vec = deriv_vec - derivs_last;
            double curvature = arma::dot(s_vec, y_vec);
            std::cout << "Curvature: " << curvature << std::endl;
            if (curvature > 0) {
                s_map[i-1] = s_vec;
                y_map[i-1] = y_vec;
            } else {
                // Skip!
                std::cout << "Curvature condition not satisfied; skipping update" << std::endl;
            }
            
            if (s_map.size() == 0) {
                
                // Just SGD with small LR
                update_vec = - 1e-10 * deriv_vec;
                
            } else {
                
                // Get update
                int idx_start = std::max(i-1,0);
                int idx_end = std::max(i-m,0);
                std::map <int, double> rho;
                std::map <int, double> alpha;
                
                // self.log.debug("   Loop #1 range: %s" % list(reversed(range(idx_end, idx_start))))
                
                double eps=1e-12;
                
                arma::vec q = deriv_vec;
                for (int i=idx_start; i>=idx_end; i--) {
                    rho[i] = 1.0 / ( arma::as_scalar(arma::dot(y_map.at(i), s_map.at(i))) + eps);
                    alpha[i] = rho.at(i) * arma::dot(s_map.at(i), q);
                    q = q - alpha.at(i) * y_map.at(i);
                }
                
                double gamma_k = arma::as_scalar(arma::dot(s_map.at(i-1), y_map.at(i-1))) / ( arma::as_scalar(arma::dot(y_map.at(i-1), y_map.at(i-1))) + eps);
                
                arma::vec z = gamma_k * q;

                idx_start = std::max(i-m,0);
                idx_end = i;
                // self.log.debug("   Loop #2 range: %s" % list(range(idx_start, idx_end)))
                for (int i=idx_start; i<idx_end; i++) {
                    double beta_i = rho.at(i) * arma::as_scalar(arma::dot(y_map.at(i),z));
                    z = z + s_map.at(i) * (alpha.at(i) - beta_i);
                }
                update_vec = -z;
            }
        }
        
        // Remove old
        int idx_remove = i - m - 1;
        if (idx_remove >= 0) {
            s_map.erase(idx_remove);
            y_map.erase(idx_remove);
        }
        
        // Advance
        vals_last = _mat_to_vec(prec_mat_curr);
        derivs_last = deriv_vec;
        
        // Update
        double step_size = _get_step_size_armijo_backtrack(prec_mat_curr, cov_mat_true, update_vec, deriv_vec);
        // double step_size = _get_step_size_wolfe_search(prec_mat_curr, cov_mat_true, update_vec, deriv_vec);
        
        std::cout << "Opt step: " << i << " / " << no_opt_steps << " - step size: " << step_size << " - Obj func: " << obj_func_new << std::endl;
        
        prec_mat_curr += step_size * _vec_to_mat(update_vec);
        
        // Advance obj func
        obj_func_prev = obj_func_new;
    }
    
    return prec_mat_curr;
}

};


