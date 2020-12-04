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
    
    int n_rows = 5;
    int n_cols = 5;
    Optimizer opt(n_rows, n_cols, idx_pairs_free);
    
    double lr = 1e-3;
    arma::mat prec_mat_init = 0.01 * arma::eye(n_rows, n_cols);
    arma::mat prec_mat_solved = opt.solve_adam(cov_mat_true, prec_mat_init, lr, 500000);
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
    
    return 0;
}
