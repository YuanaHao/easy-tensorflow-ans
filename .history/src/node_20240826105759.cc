#include <cstddef>
#include "easytf.h"
#include <numeric>
#include <functional>



node_t::node_t(vector<size_t> shape) : shape(std::move(shape)) {
    data.resize(std::accumulate(this->shape.begin(),this->shape.end(),1,std::multiplies<size_t>()));
}

auto node_t::forward() -> void {
    //TODO();
   for (auto& input : inputs) {
        input->forward();
        data=inputs[0]->data;
        shape=inputs[0]->shape;
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

