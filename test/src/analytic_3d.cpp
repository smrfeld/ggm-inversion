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
    idx_pairs_free.push_back(std::make_pair(0, 2));
    idx_pairs_free.push_back(std::make_pair(1, 2));

    arma::mat cov_mat_true = {
        {100, 0, 10},
        {0, 80, 30},
        {10, 30, 50}
    };
    
    AnalyticSolver solver(3, idx_pairs_free);
    
    auto pr = solver.solve(cov_mat_true, arma::mat());
    arma::mat prec_mat_solved = pr.second;

    std::cout << "Prec mat soln" << std::endl;
    std::cout << prec_mat_solved << std::endl;
    
    std::cout << "Cov mat target" << std::endl;
    std::cout << cov_mat_true << std::endl;
    
    std::cout << "Inverse of prec mat validate" << std::endl;
    std::cout << arma::inv(prec_mat_solved) << std::endl;
    
    return 0;
}
