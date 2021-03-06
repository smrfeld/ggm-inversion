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

    std::vector<std::pair<int,int>> idx_pairs_non_free;
    idx_pairs_non_free.push_back(std::make_pair(0, 1));
    idx_pairs_non_free.push_back(std::make_pair(0, 2));
    idx_pairs_non_free.push_back(std::make_pair(0, 4));
    idx_pairs_non_free.push_back(std::make_pair(1, 3));
    idx_pairs_non_free.push_back(std::make_pair(1, 4));
    idx_pairs_non_free.push_back(std::make_pair(2, 3));

    arma::mat cov_mat_true = {
        {100, 0, 0, 20, 0},
        {0, 80, 30, 0, 0},
        {0, 30, 6, 0, 8},
        {20, 0, 0, 40, 10},
        {0, 0, 8, 10, 60}
    };
    
    NewtonsMethodv3 nm(5, idx_pairs_free, idx_pairs_non_free);
    
    arma::mat prec_mat_init = 0.03 * arma::eye(5,5);
    nm.no_opt_steps = 5e3;
    nm.options.log_progress = true;
    nm.options.log_interval = 1;
    nm.options.log_mats = true;
    nm.options.write_interval = nm.no_opt_steps / 100;
    nm.options.write_progress = true;
    nm.options.write_dir = "../output/test_5d_newtons_method_v3/data/";
    ensure_dir_exists(nm.options.write_dir);
    auto pr = nm.solve(cov_mat_true, prec_mat_init);
    arma::mat prec_mat_solved = pr.first;
    arma::mat cov_mat_solved = pr.second;

    // report_results(prec_mat_solved, cov_mat_true, idx_pairs_free, nm);

    return 0;
}
