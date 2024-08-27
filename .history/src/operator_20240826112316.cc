#include "operator.h"
#include <cmath>
#include "easytf.h"
#include "node.h"
#include "tensor.h"

operator_t::operator_t(const vector<node_t*>& inputs) {
    assert(!inputs.empty());
    for (auto& input : inputs) {
        assert(input != nullptr);
        this->inputs.push_back(input);
    }
}

auto operator_t::forward() -> void {
    node_t::forward();
    tensor_t result = apply();
    data = result.get_data();
}


auto operator_t::backward(tensor_t&& grad, float_t lr) -> void { // tensor_t&& grad is the right value reference
    // TODO(): base on the basic class, you will get more ideas
    //TODO();
    auto grads = gradient(std::move(grad));
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i]->backward(std::move(grads[i]), lr);
    }
};