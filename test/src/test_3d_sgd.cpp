#include <iostream>
#include <vector>
#include <map>
#include <ggm_inversion>

#include "spdlog/spdlog.h"
#include <exception>
#include <armadillo>

using namespace std;
using namespace ggm;

int main() {
    
    std::vector<std::pair<int,int>> idx_pairs_free;
    idx_pairs_free.push_back(std::make_pair(0, 0));
    idx_pairs_free.push_back(std::make_pair(1, 1));
    idx_pairs_free.push_back(std::make_pair(2, 2));
    idx_pairs_free.push_back(std::make_pair(0, 1));
    idx_pairs_free.push_back(std::make_pair(1, 2));

    arma::mat cov_mat_true = {
        {100, 10, 0},
        {10, 80, 30},
        {0, 30, 50}
    };
    
    int n_rows = 3;
    int n_cols = 3;
    Optimizer opt(n_rows, n_cols, idx_pairs_free);
    
    double lr = 1e-9;
    arma::mat prec_mat_init = 0.01 * arma::eye(n_rows, n_cols);
    arma::mat prec_mat_solved = opt.solve_sgd(cov_mat_true, prec_mat_init, lr, 500);
    
    std::cout << "Solution:" << std::endl;
    std::cout << prec_mat_solved << std::endl;
    std::cout << "Cov mat:" << std::endl;
    std::cout << arma::inv(prec_mat_solved) << std::endl;

    return 0;
}
