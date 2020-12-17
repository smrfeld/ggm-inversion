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
    
    NewtonsMethod nm(5, idx_pairs_free);
    
    arma::mat prec_mat_init = 0.03 * arma::eye(5,5);
    nm.no_opt_steps = 5e3;
    nm.options.log_progress = true;
    nm.options.log_interval = 1;
    nm.options.write_interval = nm.no_opt_steps / 100;
    nm.options.write_progress = true;
    nm.options.write_dir = "../output/test_5d_newtons_method/data/";
    ensure_dir_exists(nm.options.write_dir);
    arma::mat prec_mat_solved = nm.solve(cov_mat_true, prec_mat_init);

    // report_results(prec_mat_solved, cov_mat_true, idx_pairs_free, nm);

    return 0;
}
