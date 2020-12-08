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

#include "../include/ggm_inversion_bits/optimizer_base.hpp"
#include "../include/ggm_inversion_bits/helpers.hpp"

namespace ggm {

OptimizerBase::OptimizerBase(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free) {
    _idx_pairs_free = idx_pairs_free;
    _dim = dim;
    
    // Check
    for (auto pr: _idx_pairs_free) {
        assert(pr.first < _dim);
        assert(pr.second < _dim);
    }
}

OptimizerBase::OptimizerBase(const OptimizerBase& other) {
    _copy(other);
};
OptimizerBase::OptimizerBase(OptimizerBase&& other) {
    _move(other);
};
OptimizerBase& OptimizerBase::operator=(const OptimizerBase& other) {
    if (this != &other) {
        _clean_up();
        _copy(other);
    };
    return *this;
};
OptimizerBase& OptimizerBase::operator=(OptimizerBase&& other) {
    if (this != &other) {
        _clean_up();
        _move(other);
    };
    return *this;
};
OptimizerBase::~OptimizerBase()
{
    _clean_up();
};
void OptimizerBase::_clean_up() {
    // Nothing....
};

void OptimizerBase::_copy(const OptimizerBase& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};
void OptimizerBase::_move(OptimizerBase& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};

void OptimizerBase::_log_progress_if_needed(LogOptions options, int opt_step, int no_opt_steps, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_targets) const {
    if (options.log_progress) {
        if (opt_step % options.log_interval == 0) {
            std::cout << "   Inversion: " << opt_step << " / " << no_opt_steps << std::endl;
            std::cout << "   Cov mat curr: " << std::endl;
            std::cout << cov_mat_curr << std::endl;
            std::cout << "   Cov mat targets: " << std::endl;
            std::cout << cov_mat_targets << std::endl;
        }
    }
}

void OptimizerBase::_write_progress_if_needed(WritingOptions options, int opt_step, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const {
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

double OptimizerBase::_get_first_deriv_inverse_mat(const arma::mat &cov_mat_curr, int d1, int d2, int n1, int n2) const {
    double ret = 0.0;
    ret -= cov_mat_curr(n1,d1) * cov_mat_curr(n2,d2);
    if (d1 != d2) {
        ret -= cov_mat_curr(n1,d2) * cov_mat_curr(n2,d1);
    }
    
    return ret;
}

double OptimizerBase::_get_second_deriv_inverse_mat(const arma::mat &cov_mat_curr, int d1, int d2, int d3, int d4, int n1, int n2) const {
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

arma::mat OptimizerBase::_vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}
arma::vec OptimizerBase::_mat_to_vec(const arma::mat &mat) const {
    
    arma::vec vec(_idx_pairs_free.size());
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        vec(i) = mat(pr.first, pr.second);
    }
    
    return vec;
}

double OptimizerBase::get_obj_func_val(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {

    double val = 0.0;
    for (auto idx_pair: _idx_pairs_free) {
        int i = idx_pair.first;
        int j = idx_pair.second;
        
        val += pow(cov_mat_curr(i,j) - cov_mat_true(i,j), 2);
    }
    
    return val;
}

arma::mat OptimizerBase::get_deriv_mat(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
    
    arma::mat derivs = arma::zeros(_dim, _dim);
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

arma::vec OptimizerBase::get_deriv_vec(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
    arma::mat deriv_mat = get_deriv_mat(cov_mat_curr, cov_mat_true);
    return _mat_to_vec(deriv_mat);
}

arma::mat OptimizerBase::get_hessian(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
    
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

};


