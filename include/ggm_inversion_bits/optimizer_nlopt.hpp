//
/*
File: optimizer_nlopt.hpp
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

#include "optimizer_base.hpp"

#include <nlopt.hpp>

#ifndef OPTIMIZER_NLOPT_H
#define OPTIMIZER_NLOPT_H

namespace ggm {

class OptimizerNLOpt;

struct InputObjFuncVal {
    const OptimizerNLOpt *optimizer_nlopt;
    arma::mat cov_mat_true;
};

double nlopt_obj_func(const std::vector<double> &prec_mat_std_vec, std::vector<double> &deriv_std_vec, void* input_obj_func_val);

class OptimizerNLOpt : public OptimizerBase {
private:
            
public:
    
    // Algorithm
    nlopt::algorithm algorithm = nlopt::algorithm::LD_LBFGS_NOCEDAL;
    
    // Tolerance
    double tol = 1e-10;
    
    arma::mat std_vec_to_mat(const std::vector<double> &vec) const;
    std::vector<double> mat_to_std_vec(const arma::mat &mat) const;
    
    using OptimizerBase::OptimizerBase;
        
    arma::mat solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, int no_opt_steps, LogOptions log_options=LogOptions(), WritingOptions writing_options=WritingOptions()) const override;
};

}

#endif
