#include <cstddef>
#include "easytf.h"
#include <numeric>
#include <functional>



node_t::node_t(vector<size_t> shape) : shape(std::move(shape)) {
    size_t size = 1;
    for (auto dim : this->shape) {
        size *= dim;
    }
    data.resize(size);
}

auto node_t::forward() -> void {
    if (!visited) {
        visited = true;
        for (auto& input : inputs) {
            input->forward();
        }
    }
}

auto node_t::backward(tensor_t&& grad, float_t lr) -> void {
    for (size_t i = 0; i < grad.data.size(); i++) {
        data[i] -= lr * grad.data[i];
    }
    for (auto& input : inputs) {
        input->backward(std::move(grad), lr);
    }
    visited = false;
}

