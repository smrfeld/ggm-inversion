//
/*
File: analytic.hpp
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

#include "solver_base.hpp"

#include <string>
#include <armadillo>

#ifndef ANALYTIC_H
#define ANALYTIC_H

namespace ginv {

std::pair<arma::mat, arma::mat> solve_3d_v1(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init);

std::pair<arma::mat, arma::mat> solve_no_non_free(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init);

struct AnalyticallySolvable {
    int id;
    int dim;
    std::vector<std::pair<int,int>> idx_pairs_free;
    
    bool check_matches(int dim_other, const std::vector<std::pair<int,int>>& idx_pairs_free_other) const;
    
    std::pair<arma::mat, arma::mat> (*solve)(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init);
};

bool check_idx_pairs_contains(const std::vector<std::pair<int,int>> &idx_pairs, const std::pair<int,int> &pair);

std::vector<AnalyticallySolvable> get_analytically_solvable_models(int dim);

class AnalyticSolver : public SolverBase {
                    
private:
    
    AnalyticallySolvable _solvable;
    
    /// Internal clean up
    void _clean_up();
    /// Internal copy
    void _copy(const AnalyticSolver& other);
    /// Internal move
    void _move(AnalyticSolver &other);

public:
    
    AnalyticSolver(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free);
    
    std::pair<arma::mat, arma::mat> solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const override;
};

}

#endif
