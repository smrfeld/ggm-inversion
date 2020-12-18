#include <iostream>
#include <vector>
#include <map>
#include <ggm_inversion>

#include "spdlog/spdlog.h"
#include <exception>
#include <armadillo>

using namespace std;
using namespace ginv;

void report_results(
                    const arma::mat &prec_mat_solved,
                    const arma::mat &cov_mat_true,
                    const std::vector<std::pair<int,int>> &idx_pairs_free,
                    const L2OptimizerBase &opt
                    ) {
    arma::mat cov_mat_solved = arma::inv(prec_mat_solved);
    
    std::cout << "Solution:" << std::endl;
    std::cout << prec_mat_solved << std::endl;
    std::cout << "Cov mat:" << std::endl;
    std::cout << cov_mat_solved << std::endl;
    std::cout << "Obj func: " << opt.get_obj_func_val(cov_mat_solved, cov_mat_true) << std::endl;

    // Max err
    double max = 0;
    std::pair<int,int> idxs_max;
    double max_percent = 0;
    std::pair<int,int> idxs_max_percent;
    for (auto idx_pair: idx_pairs_free) {
        int i = idx_pair.first;
        int j = idx_pair.second;
        
        double max_0 = abs(cov_mat_true(i,j) - cov_mat_solved(i,j));
        if (max_0 > max) {
            max = max_0;
            idxs_max = idx_pair;
        }
        
        double max_percent_0 = abs((cov_mat_true(i,j) - cov_mat_solved(i,j)) / cov_mat_true(i,j));
        if (max_percent_0 > max_percent) {
            max_percent = max_percent_0;
            idxs_max_percent = idx_pair;
        }
    }
    
    std::cout << "Max err: " << max << " idx pair " << idxs_max.first << " " << idxs_max.second << std::endl;
    std::cout << "Max percent: " << max_percent << " idx pair " << idxs_max_percent.first << " " << idxs_max_percent.second << std::endl;
}
