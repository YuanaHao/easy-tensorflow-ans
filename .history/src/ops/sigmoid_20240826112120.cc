#include <cassert>
#include <cmath>
#include <cstddef>
#include "easytf.h"
#include "tensor.h"

sigmoid::sigmoid(const vector<node_t*>& inputs) : operator_t(inputs) {}

auto sigmoid::apply() -> tensor_t {
    assert(inputs.size() == 1);
    const tensor_t* input = static_cast<tensor_t*>(inputs[0]);
    tensor_t result(input->get_shape(), vector<float> (input->get_data().size()));
    for (size_t i = 0; i < input->get_data().size(); ++i) {
        result.data[i] = 1.0f / (1.0f + std::exp(-input->get_data()[i]));
    }
    return result;
}

auto sigmoid::gradient(tensor_t&& grad) -> vector<tensor_t> {
    // TODO(): give your solution here

    //TODO();
    auto input = dynamic_cast<const tensor_t*>(inputs[0]);
    auto output = tensor_t(input->get_shape());
   
    for (size_t i = 0; i < input->data.size(); ++i) {
        float_t sigmoid_x = 1.0 / (1.0 + std::exp(-input->data[i]));
        output.data[i] = grad.data[i] * sigmoid_x * (1 - sigmoid_x);
    }
   
    return {std::move(output)};
}

auto sigmoid::get_shape() -> vector<size_t> {
    // TODO(): NULL
    //TODO();
        return inputs[0]->shape;
}