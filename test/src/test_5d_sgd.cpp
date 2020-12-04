#include <iostream>
#include <vector>
#include <map>
#include <ggm_inversion>

#include "spdlog/spdlog.h"
#include <exception>
#include <armadillo>

#include "common.hpp"

using namespace std;
using namespace ggm;

int main() {
    std::vector<std::pair<int,int>> idx_pairs_free;
    idx_pairs_free.push_back(std::make_pair(0, 0));
    idx_pairs_free.push_back(std::make_pair(1, 1));
    idx_pairs_free.push_back(std::make_pair(2, 2));
    idx_pairs_free.push_back(std::make_pair(3, 3));
    idx_pairs_free.push_back(std::make_pair(4, 4));

    idx_pairs_free.push_back(std::make_pair(0, 3));
    idx_pairs_free.push_back(std::make_pair(1, 2));
    idx_pairs_free.push_back(std::make_pair(2, 4));
    idx_pairs_free.push_back(std::make_pair(3, 4));

    arma::mat cov_mat_true = {
        {100, 0, 0, 20, 0},
        {0, 80, 30, 0, 0},
        {0, 30, 6, 0, 8},
        {20, 0, 0, 40, 10},
        {0, 0, 8, 10, 60}
    };
    
    int n_rows = 5;
    int n_cols = 5;
    Optimizer opt(n_rows, n_cols, idx_pairs_free);
    
    double lr = 1e-9;
    arma::mat prec_mat_init = 0.01 * arma::eye(n_rows, n_cols);
    arma::mat prec_mat_solved = opt.solve_sgd(cov_mat_true, prec_mat_init, lr, 500000);

    report_results(prec_mat_solved, cov_mat_true, idx_pairs_free, opt);

    return 0;
}
