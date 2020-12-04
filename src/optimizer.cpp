//
/*
File: optimizer.cpp
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

#include "../include/ggm_inversion_bits/optimizer.hpp"

namespace ggm {

Optimizer::Optimizer(int n_rows, int n_cols, const std::vector<std::pair<int,int>> &idx_pairs_free) {
    _idx_pairs_free = idx_pairs_free;
    _n_rows = n_rows;
    _n_cols = n_cols;
    
    // Check
    for (auto pr: _idx_pairs_free) {
        assert(pr.first < _n_rows);
        assert(pr.second < _n_cols);
    }
}

Optimizer::Optimizer(const Optimizer& other) {
    _copy(other);
};
Optimizer::Optimizer(Optimizer&& other) {
    _move(other);
};
Optimizer& Optimizer::operator=(const Optimizer& other) {
    if (this != &other) {
        _clean_up();
        _copy(other);
    };
    return *this;
};
Optimizer& Optimizer::operator=(Optimizer&& other) {
    if (this != &other) {
        _clean_up();
        _move(other);
    };
    return *this;
};
Optimizer::~Optimizer()
{
    _clean_up();
};
void Optimizer::_clean_up() {
    // Nothing....
};

void Optimizer::_copy(const Optimizer& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _n_cols = other._n_cols;
    _n_rows = other._n_rows;
};
void Optimizer::_move(Optimizer& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _n_cols = other._n_cols;
    _n_rows = other._n_rows;
};

double Optimizer::_get_first_deriv_inverse_mat(const arma::mat &cov_mat_curr, int d1, int d2, int n1, int n2) const {
    double ret = 0.0;
    ret -= cov_mat_curr(n1,d1) * cov_mat_curr(n2,d2);
    if (d1 != d2) {
        ret -= cov_mat_curr(n1,d2) * cov_mat_curr(n2,d1);
    }
    
    return ret;
}

double Optimizer::_get_second_deriv_inverse_mat(const arma::mat &cov_mat_curr, int d1, int d2, int d3, int d4, int n1, int n2) const {
    int e = d1;
    int f = d2;
    int a = d3;
    int b = d4;
    int c = n1;
    int d = n2;
    
    double ret = 0.0;
    ret -= _get_first_deriv_inverse_mat(cov_mat_curr, e, f, c, a) * cov_mat_curr(d,b);
    ret -= cov_mat_curr(c,a) * _get_first_deriv_inverse_mat(cov_mat_curr, e, f, d, b);
    if (a != b) {
        ret -= _get_first_deriv_inverse_mat(cov_mat_curr, e, f, c, b) * cov_mat_curr(d,a);
        ret -= cov_mat_curr(c,b) * _get_first_deriv_inverse_mat(cov_mat_curr, e, f, d, a);
    }
    
    return ret;
}

arma::mat Optimizer::_vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_n_rows,_n_cols);
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}
arma::vec Optimizer::_mat_to_vec(const arma::mat &mat) const {
    
    arma::vec vec(_idx_pairs_free.size());
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        vec(i) = mat(pr.first, pr.second);
    }
    
    return vec;
}

double Optimizer::_get_step_size_armijo_backtrack(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_true, const arma::vec &updates, const arma::vec &grads, double c) const {
    
    double alpha = 1.0;
    double obj_func_0 = get_obj_func_val(arma::inv(prec_mat_curr), cov_mat_true);
    
    // This had better be a descent direction
    double ax = arma::dot(grads, updates);
    assert(ax < 0);
    
    while (obj_func_0 - get_obj_func_val(arma::inv(prec_mat_curr + alpha * _vec_to_mat(updates)), cov_mat_true) < alpha * c * ax) {
        alpha *= 0.5;
    }
    
    return alpha;
}

double Optimizer::get_obj_func_val(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {

    double val = 0.0;
    for (auto idx_pair: _idx_pairs_free) {
        int i = idx_pair.first;
        int j = idx_pair.second;
        
        val += pow(cov_mat_curr(i,j) - cov_mat_true(i,j), 2);
    }
    
    return val;
}

arma::mat Optimizer::get_deriv_mat(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
    
    arma::mat derivs = arma::zeros(_n_rows, _n_cols);
    for (auto idx_pair_deriv: _idx_pairs_free) {
        int i = idx_pair_deriv.first;
        int j = idx_pair_deriv.second;
        
        double deriv = 0.0;
        for (auto idx_pair_sum: _idx_pairs_free) {
            int k = idx_pair_sum.first;
            int l = idx_pair_sum.second;
            
            deriv += 2 * (cov_mat_curr(k,l) - cov_mat_true(k,l)) * _get_first_deriv_inverse_mat(cov_mat_curr, i, j, k, l);
        }
        
        derivs(i,j) = deriv;
        derivs(j,i) = deriv;
    }
    
    return derivs;
}

arma::vec Optimizer::get_deriv_vec(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
    arma::mat deriv_mat = get_deriv_mat(cov_mat_curr, cov_mat_true);
    return _mat_to_vec(deriv_mat);
}

arma::mat Optimizer::get_hessian(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
    
    arma::mat hessian = arma::zeros(_idx_pairs_free.size(), _idx_pairs_free.size());
    for (size_t idx_1=0; idx_1<_idx_pairs_free.size(); idx_1++) {
        auto idx_pair_deriv_1 = _idx_pairs_free.at(idx_1);
        int i = idx_pair_deriv_1.first;
        int j = idx_pair_deriv_1.second;
        
        for (size_t idx_2=0; idx_2<_idx_pairs_free.size(); idx_2++) {
            auto idx_pair_deriv_2 = _idx_pairs_free.at(idx_2);
            int x = idx_pair_deriv_2.first;
            int y = idx_pair_deriv_2.second;
            
            double deriv = 0.0;
            for (auto idx_pair_sum: _idx_pairs_free) {
                int k = idx_pair_sum.first;
                int l = idx_pair_sum.second;
                                
                deriv += 2 * _get_first_deriv_inverse_mat(cov_mat_curr, x, y, k, l) * _get_first_deriv_inverse_mat(cov_mat_curr, i, j, k, l);
                deriv += 2 * (cov_mat_curr(k,l) - cov_mat_true(k,l)) * _get_second_deriv_inverse_mat(cov_mat_curr, x, y, i, j, k, l);
            }
            
            hessian(idx_1, idx_2) = deriv;
        }
    }
    
    return hessian;
}

arma::mat Optimizer::solve_sgd(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, double lr, int no_opt_steps) const {
            
    arma::mat prec_mat_curr = prec_mat_init;
        
    for (size_t i=0; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
                    
        arma::mat derivs = get_deriv_mat(cov_mat_curr, cov_mat_true);
        prec_mat_curr -= lr * derivs;
    }
    
    return prec_mat_curr;
}

arma::mat Optimizer::solve_adam(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, double lr, int no_opt_steps, double adam_beta_1, double adam_beta_2, double adam_eps) const {
            
    arma::mat prec_mat_curr = prec_mat_init;
    
    arma::mat adam_mt, adam_vt;
    
    for (size_t i=0; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
                    
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

arma::mat Optimizer::solve_l_bfgs(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, int no_opt_steps, int m, double tol) const {
 
    arma::mat prec_mat_curr = prec_mat_init;
    std::map<int,arma::vec> s_map, y_map;
    arma::vec derivs_last, vals_last;
    double obj_func_0 = 1e16;
    
    for (int i=0; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
        
        // Check convergence
        double obj_func_1 = get_obj_func_val(cov_mat_curr, cov_mat_true);
        if (abs(obj_func_1 - obj_func_0) < tol) {
            std::cout << "Converged: obj func val: " << obj_func_1 << " prev: " << obj_func_0 << " change: " << abs(obj_func_1 - obj_func_0) << " < " << tol << std::endl;
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
            s_map[i-1] = _mat_to_vec(prec_mat_curr) - vals_last;
            y_map[i-1] = deriv_vec - derivs_last;
            
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
        double step_size = _get_step_size_armijo_backtrack(cov_mat_curr, cov_mat_true, update_vec, deriv_vec);
        
        obj_func_0 = get_obj_func_val(cov_mat_curr, cov_mat_true);
        std::cout << "Opt step: " << i << " / " << no_opt_steps << " - step size: " << step_size << " - Obj func: " << obj_func_0 << std::endl;
        
        prec_mat_curr += step_size * _vec_to_mat(update_vec);
    }
    
    return prec_mat_curr;
}

};


