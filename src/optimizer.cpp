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

double Optimizer::get_obj_func_val(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {

    double val = 0.0;
    for (auto idx_pair: _idx_pairs_free) {
        int i = idx_pair.first;
        int j = idx_pair.second;
        
        val += pow(cov_mat_curr(i,j) - cov_mat_true(i,j), 2);
    }
    
    return val;
}

arma::mat Optimizer::get_derivs(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const {
    
    arma::mat derivs = arma::zeros(_n_rows, _n_cols);
    for (auto idx_pair: _idx_pairs_free) {
        int i = idx_pair.first;
        int j = idx_pair.second;
        
        double t1 = cov_mat_curr(i,j) - cov_mat_true(i,j);
        double t2 = cov_mat_curr(i,i) * cov_mat_curr(j,j);
        if (i != j) {
            t2 += pow(cov_mat_curr(i,j), 2);
        }
        
        derivs(i,j) = -2.0 * t1 * t2;
        derivs(j,i) = derivs(i,j);
    }
    
    return derivs;
}

arma::mat Optimizer::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, double lr, int no_opt_steps) const {
            
    arma::mat prec_mat_curr = prec_mat_init;
        
    for (size_t i=0; i<no_opt_steps; i++) {
        arma::mat cov_mat_curr = arma::inv(prec_mat_curr);
                    
        arma::mat derivs = get_derivs(cov_mat_curr, cov_mat_true);
        prec_mat_curr -= lr * derivs;
    }
    
    return prec_mat_curr;
}

};


