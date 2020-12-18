//
/*
File: root_finding_newton.hpp
Created by: Oliver K. Ernst
Date: 5/27/20

MIT License

Copyright (c) 2020 Oliver K. Ernst

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "solver_base.hpp"

#include <string>
#include <armadillo>

#ifndef NEWTONS_METHOD_H
#define NEWTONS_METHOD_H

namespace ginv {

class RootFindingNewton : public SolverBase {
        
protected:
            
    bool _check_convergence(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const;
    
    void _log_progress_if_needed(Options options, int opt_step, int no_opt_steps, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_targets, const arma::mat &prec_mat_curr) const;
    
    void _write_progress_if_needed(Options options, int opt_step, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const;
    
private:
    
    /// Internal clean up
    void _clean_up();
    /// Internal copy
    void _copy(const RootFindingNewton& other);
    /// Internal move
    void _move(RootFindingNewton &other);

public:
    
    double conv_max_abs_res = 0.01;
    double conv_mean_abs_res = 0.01;
    int conv_max_no_opt_steps = 100;
    Options options;
    
    using SolverBase::SolverBase;
    
    arma::mat get_i_mat(int k, int l) const;
    arma::vec upper_tri_to_vec(const arma::mat &mat) const;
    
    arma::vec get_residuals(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const;
    arma::mat get_jacobian(const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr) const;

    std::pair<arma::mat,arma::mat> solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const override;
};

}

#endif
