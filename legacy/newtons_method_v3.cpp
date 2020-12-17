//
/*
File: newtons_method_v3.cpp
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

#include "../include/ggm_inversion_bits/newtons_method_v3.hpp"
#include "../include/ggm_inversion_bits/helpers.hpp"

namespace ggm {

NewtonsMethodv3::NewtonsMethodv3(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free, const std::vector<std::pair<int,int>> &idx_pairs_non_free) {
    _idx_pairs_free = idx_pairs_free;
    _idx_pairs_non_free = idx_pairs_non_free;
    _dim = dim;
    
    // Check
    for (auto pr: _idx_pairs_free) {
        assert(pr.first < _dim);
        assert(pr.second < _dim);
    }
}

NewtonsMethodv3::NewtonsMethodv3(const NewtonsMethodv3& other) {
    _copy(other);
};
NewtonsMethodv3::NewtonsMethodv3(NewtonsMethodv3&& other) {
    _move(other);
};
NewtonsMethodv3& NewtonsMethodv3::operator=(const NewtonsMethodv3& other) {
    if (this != &other) {
        _clean_up();
        _copy(other);
    };
    return *this;
};
NewtonsMethodv3& NewtonsMethodv3::operator=(NewtonsMethodv3&& other) {
    if (this != &other) {
        _clean_up();
        _move(other);
    };
    return *this;
};
NewtonsMethodv3::~NewtonsMethodv3()
{
    _clean_up();
};
void NewtonsMethodv3::_clean_up() {
    // Nothing....
};

void NewtonsMethodv3::_copy(const NewtonsMethodv3& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};
void NewtonsMethodv3::_move(NewtonsMethodv3& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};

void NewtonsMethodv3::_log_progress_if_needed(Options options, int opt_step, int no_opt_steps, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_targets, const arma::mat &prec_mat_curr) const {
    if (options.log_progress) {
        if (opt_step % options.log_interval == 0) {
            
            // Measure errors
            /*
            arma::vec cov_vec_targets = mat_to_vec(cov_mat_targets);
            arma::vec cov_vec_curr = mat_to_vec(cov_mat_curr);
            arma::vec cov_vec_perc_err = abs(cov_vec_curr - cov_vec_targets) / abs(cov_vec_curr);
            double ave_err = arma::mean(cov_vec_perc_err);
            double max_err = arma::max(cov_vec_perc_err);
            
            std::cout << "   Inversion: " << opt_step << " / " << no_opt_steps << " ave err: " << 100.0*ave_err << "% max err: " << 100.0*max_err << "%" << std::endl;
             */
            if (options.log_mats) {
                std::cout << "   Cov mat curr: " << std::endl;
                std::cout << cov_mat_curr << std::endl;
                std::cout << "   Prec mat curr: " << std::endl;
                std::cout << prec_mat_curr << std::endl;
                std::cout << "   Inv of current prec mat: " << std::endl;
                std::cout << arma::inv(prec_mat_curr) << std::endl;
                
                std::cout << "   Current eqs:" << std::endl;
                std::cout << get_eqs(prec_mat_curr, cov_mat_curr) << std::endl;
            }
        }
    }
}

void NewtonsMethodv3::_write_progress_if_needed(Options options, int opt_step, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
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

arma::mat NewtonsMethodv3::get_i_mat(int k, int l) const {
    arma::mat x = arma::zeros(_dim, _dim);
    x(k,l) = 1;
    x(l,k) = 1;
    return x;
}

arma::vec NewtonsMethodv3::upper_tri_to_vec(const arma::mat &mat) const {
    int no_dofs = (_dim * (_dim + 1) ) / 2;
    
    arma::vec vec(no_dofs);
    int x = 0;
    for (auto i=0; i<_dim; i++) {
        for (auto j=i; j<_dim; j++) {
            vec(x) = mat(i,j);
            x++;
        }
    }
    
    return vec;
}

arma::vec NewtonsMethodv3::get_eqs(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const {
    arma::mat tmp = arma::inv(prec_mat_curr*cov_mat_curr) - arma::eye(_dim,_dim);
    return upper_tri_to_vec(tmp);
}

arma::mat NewtonsMethodv3::get_jacobian(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const {
    
    arma::mat prec_mat_curr_inv = arma::inv(prec_mat_curr);
    arma::mat cov_mat_curr_inv = arma::inv(cov_mat_curr);
    
    int no_dofs = (_dim * (_dim + 1) ) / 2;
    arma::mat jac(no_dofs, no_dofs);
    
    // First: derivs wrt free elements of B
    for (auto i_dof=0; i_dof<_idx_pairs_free.size(); i_dof++) {
        int k = _idx_pairs_free.at(i_dof).first;
        int l = _idx_pairs_free.at(i_dof).second;
        
        arma::mat deriv_f_wrt_bkl = - cov_mat_curr_inv * prec_mat_curr_inv * get_i_mat(k, l) * prec_mat_curr_inv;
        jac.col(i_dof) = upper_tri_to_vec(deriv_f_wrt_bkl);
    }
    
    // Second: derivs wrt non-free elements of Sigma
    for (auto j_dof=0; j_dof<_idx_pairs_non_free.size(); j_dof++) {
        int i_dof = j_dof + _idx_pairs_free.size();
        
        int k = _idx_pairs_non_free.at(j_dof).first;
        int l = _idx_pairs_non_free.at(j_dof).second;

        arma::mat deriv_f_wrt_skl = - cov_mat_curr_inv * get_i_mat(k, l) * cov_mat_curr_inv * prec_mat_curr_inv;
        jac.col(i_dof) = upper_tri_to_vec(deriv_f_wrt_skl);
    }
    
    return jac;
}

arma::mat NewtonsMethodv3::free_vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}

arma::mat NewtonsMethodv3::non_free_vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_non_free.size(); i++) {
        auto pr = _idx_pairs_non_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}

std::pair<arma::mat,arma::mat> NewtonsMethodv3::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const {
    
    arma::mat prec_mat_curr = prec_mat_init;
    arma::mat cov_mat_curr = cov_mat_true;
        
    for (size_t i=0; i<no_opt_steps; i++) {
        
        // Log if needed
        _log_progress_if_needed(options, i, no_opt_steps, cov_mat_curr, cov_mat_true, prec_mat_curr);
        
        // Write if needed
        _write_progress_if_needed(options, i, prec_mat_curr, cov_mat_curr, cov_mat_true);
        
        // Update
        arma::vec f = get_eqs(prec_mat_curr, cov_mat_curr);
        arma::mat jac = get_jacobian(prec_mat_curr, cov_mat_curr);
        arma::vec update_vec = arma::solve(jac, - f);

        // Update
        arma::vec update_vec_b = update_vec.subvec(0, _idx_pairs_free.size()-1);
        arma::vec update_vec_sigma = update_vec.subvec(_idx_pairs_free.size(), update_vec.n_rows-1);
        arma::mat update_mat_b = free_vec_to_mat(update_vec_b);
        arma::mat update_mat_sigma = non_free_vec_to_mat(update_vec_sigma);
        
        prec_mat_curr += update_mat_b;
        cov_mat_curr += update_mat_sigma;
    }
    
    return std::make_pair(prec_mat_curr,cov_mat_curr);
}

};


