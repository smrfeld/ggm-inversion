//
/*
File: root_finding_newton.cpp
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

#include "../include/ggm_inversion_bits/root_finding_newton.hpp"
#include "../include/ggm_inversion_bits/helpers.hpp"

namespace ginv {

bool RootFindingNewton::_check_pair_exists(const std::vector<std::pair<int,int>> &pairs, std::pair<int,int> pr_search) const {
                
    auto it = std::find(pairs.begin(), pairs.end(), pr_search);
    if (it != pairs.end()) {
        return true;
    }
    
    std::pair<int,int> pr_reverse = std::make_pair(pr_search.second, pr_search.first);
    auto it2 = std::find(pairs.begin(), pairs.end(), pr_reverse);
    if (it2 != pairs.end()) {
        return true;
    } else {
        return false;
    }
}

RootFindingNewton::RootFindingNewton(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free) : SolverBase(dim, idx_pairs_free) {
    
    // Non-free idxs
    for (auto i=0; i<_dim; i++) {
        for (auto j=i; j<_dim; j++) {
            std::pair<int,int> pr = std::make_pair(i,j);
            if (!_check_pair_exists(_idx_pairs_free, pr)) {
                _idx_pairs_non_free.push_back(pr);
            }
        }
    }
}

RootFindingNewton::RootFindingNewton(const RootFindingNewton& other) : SolverBase(other) {
    _copy(other);
};
RootFindingNewton::RootFindingNewton(RootFindingNewton&& other) : SolverBase(std::move(other)) {
    _move(other);
};
RootFindingNewton& RootFindingNewton::operator=(const RootFindingNewton& other) {
    if (this != &other) {
        _clean_up();
        SolverBase::operator=(other);
        _copy(other);
    };
    return *this;
};
RootFindingNewton& RootFindingNewton::operator=(RootFindingNewton&& other) {
    if (this != &other) {
        _clean_up();
        SolverBase::operator=(std::move(other));
        _move(other);
    };
    return *this;
};
RootFindingNewton::~RootFindingNewton()
{
    _clean_up();
};
void RootFindingNewton::_clean_up() {
    // Nothing....
};

void RootFindingNewton::_copy(const RootFindingNewton& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};
void RootFindingNewton::_move(RootFindingNewton& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};

void RootFindingNewton::_log_progress_if_needed(Options options, int opt_step, int no_opt_steps, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_targets, const arma::mat &prec_mat_curr) const {
    if (options.log_progress) {
        if (opt_step % options.log_interval == 0) {
            
            // Measure residuals
            arma::vec res = get_residuals(prec_mat_curr, cov_mat_curr);
            double max_res = arma::max(res);
            double mean_res = arma::mean(res);
            
            std::cout << "   Inversion: " << opt_step << " / " << no_opt_steps << " ave residual: " << mean_res << " max residual: " << max_res << std::endl;

            if (options.log_mats) {
                std::cout << "   Cov mat curr: " << std::endl;
                std::cout << cov_mat_curr << std::endl;
                std::cout << "   Inv of current prec mat: " << std::endl;
                std::cout << arma::inv(prec_mat_curr) << std::endl;
                std::cout << "   Prec mat curr: " << std::endl;
                std::cout << prec_mat_curr << std::endl;
            }
        }
    }
}

void RootFindingNewton::_write_progress_if_needed(Options options, int opt_step, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const {
    if (options.write_progress) {
        assert (options.write_dir != "");
        
        if (opt_step % options.write_interval == 0) {
            // Write
            std::string fname = options.write_dir + "prec_mat.txt";
            write_mat(fname, opt_step, opt_step!=0, prec_mat_curr);
            
            fname = options.write_dir + "cov_mat.txt";
            write_mat(fname, opt_step, opt_step!=0, cov_mat_curr);
        }
    }
}

arma::mat RootFindingNewton::get_i_mat(int k, int l) const {
    arma::mat x = arma::zeros(_dim, _dim);
    x(k,l) = 1;
    x(l,k) = 1;
    return x;
}

arma::vec RootFindingNewton::upper_tri_to_vec(const arma::mat &mat) const {
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

arma::vec RootFindingNewton::get_residuals(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const {
    arma::mat tmp = prec_mat_curr*cov_mat_curr - arma::eye(_dim,_dim);
    return upper_tri_to_vec(tmp);
}

arma::mat RootFindingNewton::get_jacobian(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const {
    
    int no_dofs = (_dim * (_dim + 1) ) / 2;
    arma::mat jac(no_dofs, no_dofs);
    
    // First: derivs wrt free elements of B
    for (auto i_dof=0; i_dof<_idx_pairs_free.size(); i_dof++) {
        int k = _idx_pairs_free.at(i_dof).first;
        int l = _idx_pairs_free.at(i_dof).second;
        
        arma::mat deriv_f_wrt_bkl = get_i_mat(k, l) * cov_mat_curr;
        jac.col(i_dof) = upper_tri_to_vec(deriv_f_wrt_bkl);
    }
    
    // Second: derivs wrt non-free elements of Sigma
    for (auto j_dof=0; j_dof<_idx_pairs_non_free.size(); j_dof++) {
        int i_dof = j_dof + _idx_pairs_free.size();
        
        int k = _idx_pairs_non_free.at(j_dof).first;
        int l = _idx_pairs_non_free.at(j_dof).second;

        arma::mat deriv_f_wrt_skl = prec_mat_curr * get_i_mat(k, l);
        jac.col(i_dof) = upper_tri_to_vec(deriv_f_wrt_skl);
    }
    
    return jac;
}

arma::mat RootFindingNewton::free_vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}

arma::mat RootFindingNewton::non_free_vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_non_free.size(); i++) {
        auto pr = _idx_pairs_non_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}

bool RootFindingNewton::_check_convergence(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const {

    arma::vec residuals = get_residuals(prec_mat_curr, cov_mat_curr);
    
    // Max/mean
    double max_abs_res = arma::max(abs(residuals));
    double mean_abs_res = arma::mean(abs(residuals));
    
    if (max_abs_res < conv_max_abs_res) {
        if (options.log_progress) {
            std::cout << "Converged: max absolute residual: " << max_abs_res << " is less than limit: " << conv_max_abs_res << std::endl;
        }
        return true;
    }

    if (mean_abs_res < conv_mean_abs_res) {
        if (options.log_progress) {
            std::cout << "Converged: mean absolute residual: " << mean_abs_res << " is less than limit: " << conv_mean_abs_res << std::endl;
        }
        return true;
    }

    return false;
}

std::pair<arma::mat,arma::mat> RootFindingNewton::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const {
    
    arma::mat prec_mat_curr = prec_mat_init;
    arma::mat cov_mat_curr = cov_mat_true;
        
    for (size_t i=0; i<conv_max_no_opt_steps; i++) {
        
        // Check convergence
        if (_check_convergence(prec_mat_curr, cov_mat_curr)) {
            return std::make_pair(cov_mat_curr,prec_mat_curr);
        }
        
        // Log if needed
        _log_progress_if_needed(options, i, conv_max_no_opt_steps, cov_mat_curr, cov_mat_true, prec_mat_curr);
        
        // Write if needed
        _write_progress_if_needed(options, i, prec_mat_curr, cov_mat_curr);
        
        // Update
        arma::vec f = get_residuals(prec_mat_curr, cov_mat_curr);
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
    
    if (options.log_progress) {
        std::cout << "Converged: max no opt steps reached: " << conv_max_no_opt_steps << std::endl;
    }
    
    return std::make_pair(cov_mat_curr,prec_mat_curr);
}

};


