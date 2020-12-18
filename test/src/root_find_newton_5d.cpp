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
        {0, 80, 30, 0, 0},
        {0, 30, 6, 0, 8},
        {20, 0, 0, 40, 10},
        {0, 0, 8, 10, 60}
    };
    
    RootFindingNewton rfn(5, idx_pairs_free);
    
    arma::mat prec_mat_init = 0.03 * arma::eye(5,5);
    rfn.conv_max_no_opt_steps = 10;
    rfn.options.log_progress = true;
    rfn.options.log_interval = 1;
    rfn.options.log_mats = true;
    rfn.options.write_interval = 1;
    rfn.options.write_progress = true;
    rfn.options.write_dir = "../output/root_find_newton_5d/data/";
    ensure_dir_exists(rfn.options.write_dir);
    auto pr = rfn.solve(cov_mat_true, prec_mat_init);
    arma::mat cov_mat_solved = pr.first;
    arma::mat prec_mat_solved = pr.second;

    std::cout << "Solved:" << std::endl;
    std::cout << "Prec mat:" << std::endl;
    std::cout << prec_mat_solved << std::endl;
    std::cout << "Inverse(Prec mat):" << std::endl;
    std::cout << arma::inv(prec_mat_solved) << std::endl;
    std::cout << "Cov mat:" << std::endl;
    std::cout << cov_mat_solved << std::endl;
    std::cout << "Inverse(Cov mat):" << std::endl;
    std::cout << arma::inv(cov_mat_solved) << std::endl;

    return 0;
}
