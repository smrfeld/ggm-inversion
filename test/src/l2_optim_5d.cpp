#include <iostream>
#include <vector>
#include <map>
#include <ggm_inversion>

#include "spdlog/spdlog.h"
#include <exception>
#include <armadillo>

#include "common.hpp"

using namespace std;
using namespace ginv;

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
        {0, 80, 3, 0, 0},
        {0, 3, 6, 0, 4},
        {20, 0, 0, 40, 10},
        {0, 0, 4, 10, 60}
    };
    
    L2OptimizerOptim opt(5, idx_pairs_free);
    
    arma::mat prec_mat_init = 0.01 * arma::eye(5,5);
    opt.settings.iter_max = 5e5;
    arma::mat prec_mat_solved = opt.solve(cov_mat_true, prec_mat_init);
    
    report_results(prec_mat_solved, cov_mat_true, idx_pairs_free, opt);

    return 0;
}
