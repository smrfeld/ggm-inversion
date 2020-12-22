//
/*
File: analytic.cpp
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

#include "../include/ggm_inversion_bits/analytic.hpp"
#include "../include/ggm_inversion_bits/helpers.hpp"

#include <spdlog/spdlog.h>

namespace ginv {

std::vector<AnalyticallySolvable> get_analytically_solvable_models(int dim) {
    
    std::vector<AnalyticallySolvable> ret;
    
    // No free
    int id_prev = -1;
    AnalyticallySolvable v;
    v.id = ++id_prev;
    v.dim = dim;
    for (auto i=0; i<dim; i++) {
        for (auto j=i; j<dim; j++) {
            v.idx_pairs_free.push_back(std::make_pair(i,j));
        }
    }
    v.solve = &solve_no_non_free;
    ret.push_back(v);
    
    // V1
    if (dim == 3) {
        AnalyticallySolvable v1;
        v1.id = ++id_prev;
        v1.dim = 3;
        v1.idx_pairs_free = std::vector<std::pair<int,int>>({
            std::make_pair(0,0),
            std::make_pair(0,2),
            std::make_pair(1,1),
            std::make_pair(1,2),
            std::make_pair(2,2)
        });
        v1.solve = &solve_3d_v1;
        ret.push_back(v1);
    }
    
    return ret;
};

bool check_idx_pairs_contains(const std::vector<std::pair<int,int>> &idx_pairs, const std::pair<int,int> &pair) {
    auto it = std::find(idx_pairs.begin(), idx_pairs.end(), pair);
    if (it != idx_pairs.end()) {
        return true;
    }
    
    std::pair<int,int> pair_reverse = std::make_pair(pair.second,pair.first);
    auto it2 = std::find(idx_pairs.begin(), idx_pairs.end(), pair_reverse);
    if (it2 != idx_pairs.end()) {
        return true;
    }

    return false;
}

bool AnalyticallySolvable::check_matches(int dim_other, const std::vector<std::pair<int,int>>& idx_pairs_free_other) const {

    // Check dim
    if (dim_other != dim) {
        return false;
    }
    
    // Check no matches
    if (idx_pairs_free.size() != idx_pairs_free_other.size()) {
        return false;
    }
    
    for (auto const &pair_other: idx_pairs_free_other) {
        if (!check_idx_pairs_contains(idx_pairs_free, pair_other)) {
            return false;
        }
    }
    
    return true;
}

AnalyticSolver::AnalyticSolver(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free) : SolverBase(dim, idx_pairs_free) {
    
    auto solvable_models = get_analytically_solvable_models(dim);
    
    // Check
    bool matches = false;
    for (auto const &solvable_model: solvable_models) {
        bool matches_this = solvable_model.check_matches(dim, idx_pairs_free);
        if (matches_this) {
            _solvable = solvable_model;
            matches = true;
            break;
        }
    }
    
    if (!matches) {
        throw std::invalid_argument("Given model is not supported by the analytic solver!");
    }
}

std::pair<arma::mat, arma::mat> solve_3d_v1(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) {
    
    int dim = cov_mat_true.n_rows;
    
    arma::mat cov_mat_soln = cov_mat_true;
    arma::mat prec_mat_soln = arma::zeros(dim,dim);
    
    double na = cov_mat_true(0,0);
    double nb = cov_mat_true(0,2);
    double nc = cov_mat_true(1,1);
    double nd = cov_mat_true(1,2);
    double ne = cov_mat_true(2,2);

    prec_mat_soln(0,0) = ne / ( - pow(nb,2) + na*ne );
    prec_mat_soln(0,2) = - nb / ( - pow(nb,2) + na*ne );
    prec_mat_soln(1,1) = ne / ( - pow(nd,2) + nc*ne );
    prec_mat_soln(1,2) = - nd / ( - pow(nd,2) + nc*ne );
    prec_mat_soln(2,2) = (-pow(nb,2)*pow(nd,2) + na*nc*pow(ne,2)) / (ne*(pow(nb,2)-na*ne)*(pow(nd,2)-nc*ne));
    
    // Symmetric
    prec_mat_soln(2,0) = prec_mat_soln(0,2);
    prec_mat_soln(2,1) = prec_mat_soln(1,2);
    
    cov_mat_soln(0,1) = nb*nd/ne;
    
    // Symmetric
    cov_mat_soln(1,0) = cov_mat_soln(0,1);
    
    return std::make_pair(cov_mat_soln, prec_mat_soln);
}

std::pair<arma::mat, arma::mat> solve_no_non_free(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) {
    arma::mat cov_mat_soln = cov_mat_true;
    arma::mat prec_mat_soln = arma::inv(cov_mat_soln);
    return std::make_pair(cov_mat_soln, prec_mat_soln);
}

std::pair<arma::mat, arma::mat> AnalyticSolver::solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const {
    return (*_solvable.solve)(cov_mat_true, prec_mat_init);
}

};


