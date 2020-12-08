//
/*
File: optimizer_l_bfgs.hpp
Created by: Oliver K. Ernst
Date: 12/4/20

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

#ifndef OPTIMIZER_SGD_H
#define OPTIMIZER_SGD_H

namespace ggm {

class OptimizerSGD : public OptimizerBase {
public:
    
    double lr = 1.0;
    
    using OptimizerBase::OptimizerBase;
        
    arma::mat solve(const arma::mat &cov_mat_true, const arma::mat &prec_mat_init, int no_opt_steps, Options options=Options()) const override;
};

}

#endif
