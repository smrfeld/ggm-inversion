//
/*
File: optimizer_optim.hpp
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

#define OPTIM_ENABLE_ARMA_WRAPPERS
#include <optim/optim.hpp>

#ifndef OPTIMIZER_OPTIM_H
#define OPTIMIZER_OPTIM_H

namespace ggm {

class OptimizerOptim;

struct InputObjFuncVal {
    const OptimizerOptim *optimizer;
    arma::mat cov_mat_true;
};

double optim_obj_func(const arma::vec &prec_mat_vec, arma::vec *deriv_vec, void* input_obj_func_val);

class OptimizerOptim : public OptimizerBase {
            
public:
        
    // Tolerance
    double tol = 1e-10;
        
    using OptimizerBase::OptimizerBase;

    arma::mat solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, Options options=Options()) const override;
};

}

#endif
