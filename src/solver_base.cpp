//
/*
File: solver_base.cpp
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

#include "../include/ggm_inversion_bits/solver_base.hpp"
#include "../include/ggm_inversion_bits/helpers.hpp"

#include <spdlog/spdlog.h>

namespace ginv {

SolverBase::SolverBase(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free) {
    _idx_pairs_free = idx_pairs_free;
    _dim = dim;
    
    // Check
    for (auto pr: _idx_pairs_free) {
        assert(pr.first < _dim);
        assert(pr.second < _dim);
    }
    
    // Non-free idxs
    for (auto i=0; i<_dim; i++) {
        for (auto j=i; j<_dim; j++) {
            std::pair<int,int> pr = std::make_pair(i,j);
            if (!_check_pair_exists(_idx_pairs_free, pr)) {
                _idx_pairs_non_free.push_back(pr);
            }
        }
    }
}

SolverBase::SolverBase(const SolverBase& other) {
    _copy(other);
};
SolverBase::SolverBase(SolverBase&& other) {
    _move(other);
};
SolverBase& SolverBase::operator=(const SolverBase& other) {
    if (this != &other) {
        _clean_up();
        _copy(other);
    };
    return *this;
};
SolverBase& SolverBase::operator=(SolverBase&& other) {
    if (this != &other) {
        _clean_up();
        _move(other);
    };
    return *this;
};
SolverBase::~SolverBase()
{
    _clean_up();
};
void SolverBase::_clean_up() {
    // Nothing....
};

void SolverBase::_copy(const SolverBase& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};
void SolverBase::_move(SolverBase& other) {
    _idx_pairs_free = other._idx_pairs_free;
    _dim = other._dim;
};

std::string SolverBase::_get_log_header(const Options &options, int opt_step, int max_no_opt_steps) const {
    return _get_log_header(options.log_header, opt_step, max_no_opt_steps);
}

std::string SolverBase::_get_log_header(std::string log_header, int opt_step, int max_no_opt_steps) const {
    return format_str(log_header + "[Inversion: %08d / %08d] ", opt_step, max_no_opt_steps);
}

void SolverBase::_log_mat_info(const arma::mat &mat, const Options &options, int opt_step, int max_no_opt_steps) const {
    std::string header = _get_log_header(options, opt_step, max_no_opt_steps);
    _log_mat_info(mat, header);
}

void SolverBase::_log_mat_info(const arma::mat &mat, std::string header) const {

    for (auto i_row=0; i_row<mat.n_rows; i_row++) {
        std::string row = "";
        for (auto i_col=0; i_col<mat.n_cols; i_col++) {
            row += format_str("%8.4f ", mat(i_row,i_col));
        }
        spdlog::info(header + " " + row);
    }
}

bool SolverBase::_check_pair_exists(const std::vector<std::pair<int,int>> &pairs, std::pair<int,int> pr_search) const {
                
    auto it = std::find(pairs.begin(), pairs.end(), pr_search);
    if (it != pairs.end()) {
        return true;
    }
    
    std::pair<int,int> pr_reverse = std::make_pair(pr_search.second, pr_search.first);
    auto it2 = std::find(pairs.begin(), pairs.end(), pr_reverse);
    if (it2 != pairs.end()) {
        return true;
    } else {
        return false;
    }
}

std::vector<std::pair<int,int>> SolverBase::get_idx_pairs_free() const {
    return _idx_pairs_free;
}

int SolverBase::get_dim() const {
    return _dim;
}

arma::mat SolverBase::free_vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}

arma::mat SolverBase::non_free_vec_to_mat(const arma::vec &vec) const {
    
    arma::mat mat = arma::zeros(_dim,_dim);
    for (size_t i=0; i<_idx_pairs_non_free.size(); i++) {
        auto pr = _idx_pairs_non_free.at(i);
        mat(pr.first, pr.second) = vec(i);
        mat(pr.second, pr.first) = vec(i);
    }
    
    return mat;
}

arma::vec SolverBase::free_mat_to_vec(const arma::mat &mat) const {
    
    arma::vec vec(_idx_pairs_free.size());
    for (size_t i=0; i<_idx_pairs_free.size(); i++) {
        auto pr = _idx_pairs_free.at(i);
        vec(i) = mat(pr.first, pr.second);
    }
    
    return vec;
}

arma::vec SolverBase::non_free_mat_to_vec(const arma::mat &mat) const {
    
    arma::vec vec(_idx_pairs_non_free.size());
    for (size_t i=0; i<_idx_pairs_non_free.size(); i++) {
        auto pr = _idx_pairs_non_free.at(i);
        vec(i) = mat(pr.first, pr.second);
    }
    
    return vec;
}

arma::mat SolverBase::zero_free_elements(const arma::mat &mat) const {
    arma::mat out = mat;
    for (auto pr: _idx_pairs_free) {
        out(pr.first,pr.second) = 0;
        out(pr.second,pr.first) = 0;
    }
    
    return out;
}

arma::mat SolverBase::zero_non_free_elements(const arma::mat &mat) const {
    arma::mat out = mat;
    for (auto pr: _idx_pairs_non_free) {
        out(pr.first,pr.second) = 0;
        out(pr.second,pr.first) = 0;
    }
    
    return out;
}

};


