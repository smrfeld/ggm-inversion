//
/*
File: newtons_method_v1.hpp
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

#include "options.hpp"

#include <string>
#include <armadillo>

#ifndef NEWTONS_METHOD_V1_H
#define NEWTONS_METHOD_V1_H

namespace ggm {

class NewtonsMethodv1 {
        
protected:
    
    std::vector<std::pair<int,int>> _idx_pairs_free;
    int _dim;
    
    double _get_first_deriv_inverse_mat(const arma::mat &cov_mat_curr, int d1, int d2, int n1, int n2) const;

    void _log_progress_if_needed(Options options, int opt_step, int no_opt_steps, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_targets, const arma::mat &prec_mat_curr) const;
    
    void _write_progress_if_needed(Options options, int opt_step, const arma::mat &prec_mat_curr, const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;
    
private:
    
    /// Internal clean up
    void _clean_up();
    /// Internal copy
    void _copy(const NewtonsMethodv1& other);
    /// Internal move
    void _move(NewtonsMethodv1 &other);

public:
    
    int no_opt_steps = 100;
    Options options;
    
    NewtonsMethodv1(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free);
    NewtonsMethodv1(const NewtonsMethodv1& other);
    NewtonsMethodv1& operator=(const NewtonsMethodv1& other);
    NewtonsMethodv1(NewtonsMethodv1&& other);
    NewtonsMethodv1& operator=(NewtonsMethodv1&& other);
    ~NewtonsMethodv1();

    arma::mat vec_to_mat(const arma::vec &vec) const;
    arma::vec mat_to_vec(const arma::mat &mat) const;
    
    arma::vec get_eq_vals(const arma::mat &cov_mat_curr, const arma::mat &cov_mat_true) const;
    
    arma::mat get_jacobian(const arma::mat &cov_mat_curr) const;
    
    arma::mat solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init) const;
};

}

#endif
