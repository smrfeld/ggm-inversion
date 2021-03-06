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
    
    L2OptimizerAdam opt(5, idx_pairs_free);
    
    arma::mat prec_mat_init = 0.01 * arma::eye(5,5);
    opt.lr = 1e-3;
    opt.no_opt_steps = 5e4;
    opt.options.write_interval = opt.no_opt_steps / 100;
    opt.options.write_progress = true;
    opt.options.write_dir = "../output/l2_adam_5d/data/";
    ensure_dir_exists(opt.options.write_dir);
    auto pr = opt.solve(cov_mat_true, prec_mat_init);
    arma::mat prec_mat_solved = pr.second;
    
    report_results(prec_mat_solved, cov_mat_true, idx_pairs_free, opt);

    return 0;
}
