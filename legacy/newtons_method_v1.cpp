//
/*
File: newtons_method_v1.cpp
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

#include "../include/ggm_inversion_bits/newtons_method_v1.hpp"
#include "../include/ggm_inversion_bits/helpers.hpp"

namespace ggm {

NewtonsMethodv1::NewtonsMethodv1(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free) {
    _idx_pairs_free = idx_pairs_free;
    _dim = dim;
    
    // Check
    for (auto pr: _idx_pairs_free) {
        assert(pr.first < _dim);
        assert(pr.second < _dim);
    }
}

NewtonsMethodv1::NewtonsMethodv1(const NewtonsMethodv1& other) {
    _copy(other);
};
NewtonsMethodv1::NewtonsMethodv1(NewtonsMethodv1&& other) {
    _move(other);
};
NewtonsMethodv1& NewtonsMethodv1::operator=(const NewtonsMethodv1& other) {
    if (this != &other) {
        _clean_up();
        _copy(other);
    };
    return *this;
};
NewtonsMethodv1& NewtonsMethodv1::operator=(NewtonsMethodv1&& other) {
    if (this != &other) {
        _clean_up();
        _move(other);
    };
    return *this;
};
NewtonsMethodv1::~NewtonsMethodv1()
{
    _clean_up();
};
void NewtonsMethodv1::_clean_up() {
    // Nothing....
};

void NewtonsMethodv1::_copy(const NewtonsMethodv1& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};
void NewtonsMethodv1::_move(NewtonsMethodv1& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};

void NewtonsMethodv1::_log_progress_if_needed(Options options, int opt_step, int no_opt_steps, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_targets, const arma::mat &prec_mat_curr) const {
    if (options.log_progress) {
        if (opt_step % options.log_interval == 0) {
            
            // Measure errors
            arma::vec cov_vec_targets = mat_to_vec(cov_mat_targets);
            arma::vec cov_vec_curr = mat_to_vec(cov_mat_curr);
            arma::vec cov_vec_perc_err = abs(cov_vec_curr - cov_vec_targets) / abs(cov_vec_curr);
            double ave_err = arma::mean(cov_vec_perc_err);
            double max_err = arma::max(cov_vec_perc_err);
            
            std::cout << "   Inversion: " << opt_step << " / " << no_opt_steps << " ave err: " << 100.0*ave_err << "% max err: " << 100.0*max_err << "%" << std::endl;
            if (options.log_mats) {
                std::cout << "   Cov mat curr: " << std::endl;
                std::cout << cov_mat_curr << std::endl;
                std::cout << "   Cov mat targets: " << std::endl;
                std::cout << cov_mat_targets << std::endl;
                std::cout << "   Prec mat curr: " << std::endl;
                std::cout << prec_mat_curr << std::endl;
            }
        }
    }
}

void NewtonsMethodv1::_write_progress_if_needed(Options options, int opt_step, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
    if (options.write_progress) {
        assert (options.write_dir != "");
        
        if (opt_step % options.write_interval == 0) {
            // Write
            std::string fname = options.write_dir + "prec_mat.txt";
            write_submat(fname, opt_step, opt_step!=0, prec_mat_curr, _idx_pairs_free);
            
            fname = options.write_dir + "cov_mat.txt";
            write_submat(fname, opt_step, opt_step!=0, cov_mat_curr, _idx_pairs_free);
            
            if (opt_step == 0) {
                fname = options.write_dir + "cov_mat_targets.txt";
                write_submat(fname, false, cov_mat_true, _idx_pairs_free);
            }
        }
    }
}

double NewtonsMethodv1::_get_first_deriv_inverse_mat(const arma::mat &cov_mat_curr, int d1, int d2, int n1, int n2) const {
    double ret = 0.0;
    ret -= cov_mat_curr(n1,d1) * cov_mat_curr(n2,d2);
    if (d1 != d2) {
        ret -= cov_mat_curr(n1,d2) * cov_mat_curr(n2,d1);
    }
    
    return ret;
}

arma::mat NewtonsMethodv1::vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}
arma::vec NewtonsMethodv1::mat_to_vec(const arma::mat &mat) const {
    
    arma::vec vec(_idx_pairs_free.size());
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        vec(i) = mat(pr.first, pr.second);
    }
    
    return vec;
}

arma::vec NewtonsMethodv1::get_eq_vals(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
        
    arma::vec cov_vec_curr = mat_to_vec(cov_mat_curr);
    arma::vec cov_vec_true = mat_to_vec(cov_mat_true);
        
    return cov_vec_curr - cov_vec_true;
}

arma::mat NewtonsMethodv1::get_jacobian(const arma::mat &cov_mat_curr) const {
    
    arma::mat jac(_idx_pairs_free.size(),_idx_pairs_free.size());
    
    for (auto i_num=0; i_num<_idx_pairs_free.size(); i_num++) {
        int n1 = _idx_pairs_free.at(i_num).first;
        int n2 = _idx_pairs_free.at(i_num).second;
        
        for (auto i_denom=0; i_denom<_idx_pairs_free.size(); i_denom++) {
            int d1 = _idx_pairs_free.at(i_denom).first;
            int d2 = _idx_pairs_free.at(i_denom).second;
            
            jac(i_num,i_denom) = _get_first_deriv_inverse_mat(cov_mat_curr, d1, d2, n1, n2);
        }
    }
    
    return jac;
}

arma::mat NewtonsMethodv1::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const {
    
    arma::mat prec_mat_curr = prec_mat_init;
        
    for (size_t i=0; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
                    
        // Log if needed
        _log_progress_if_needed(options, i, no_opt_steps, cov_mat_curr, cov_mat_true, prec_mat_curr);
        
        // Write if needed
        _write_progress_if_needed(options, i, prec_mat_curr, cov_mat_curr, cov_mat_true);
        
        // Solve linear equation
        arma::mat jac = get_jacobian(cov_mat_curr);
        std::cout << jac << std::endl;
        arma::vec eq_vals = get_eq_vals(cov_mat_curr, cov_mat_true);
        arma::vec update_vec = arma::solve(jac, -eq_vals);
        
        // Update
        arma::mat update_mat = vec_to_mat(update_vec);
        prec_mat_curr += update_mat;
    }
    
    return prec_mat_curr;
}

};


