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
    idx_pairs_free.push_back(std::make_pair(0, 1));
    idx_pairs_free.push_back(std::make_pair(1, 2));

    arma::mat cov_mat_true = {
        {100, 10, 0},
        {10, 80, 30},
        {0, 30, 50}
    };
    
    OptimizerGD opt(3, idx_pairs_free);
    
    arma::mat prec_mat_init = 0.01 * arma::eye(3, 3);
    opt.lr = 1e-9;
    opt.no_opt_steps = 500;
    opt.options.write_interval = opt.no_opt_steps / 100;
    opt.options.write_progress = true;
    opt.options.write_dir = "../output/test_3d_gd/data/";
    ensure_dir_exists(opt.options.write_dir);
    arma::mat prec_mat_solved = opt.solve(cov_mat_true, prec_mat_init);

    report_results(prec_mat_solved, cov_mat_true, idx_pairs_free, opt);
    
    return 0;
}
