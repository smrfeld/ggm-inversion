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

namespace ginv {

SolverBase::SolverBase(int dim, const std::vector<std::pair<int,int>> &idx_pairs_free) {
    _idx_pairs_free = idx_pairs_free;
    _dim = dim;
    
    // Check
    for (auto pr: _idx_pairs_free) {
        assert(pr.first < _dim);
        assert(pr.second < _dim);
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

};


