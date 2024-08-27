#include <cassert>
#include <cstddef>
#include "easytf.h"
#include "tensor.h"

matmul::matmul(const vector<node_t*>& inputs) : operator_t(inputs) {}

auto transpose(const tensor_t& t) -> tensor_t {
    // TODO(): Matrix transpose
    //TODO();
    auto shape = t.get_shape();
    auto new_shape = vector<size_t>{shape[1], shape[0]};
    tensor_t result(new_shape);
   
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            result.data[j * shape[0] + i] = t.get_data()[i * shape[1] + j];
        }
    }
   
    return result;
}

auto _matmul_(const tensor_t* a, const tensor_t* b) -> tensor_t {
    auto a_shape = a->get_shape();
    auto b_shape = b->get_shape();

    assert(a_shape.size() == 2);
    assert(b_shape.size() == 2);
    assert(a_shape[1] == b_shape[0]);

    auto c_shape = vector<size_t>{a_shape[0], b_shape[1]};
    auto c = tensor_t(c_shape);

    auto a_data = a->get_data();
    auto b_data = b->get_data();
    auto& c_data = c.data;

    // TODO(): Matrix Matmul
    //TODO();
    for (size_t i = 0; i < a_shape[0]; i++) {
        for (size_t j = 0; j < b_shape[1]; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a_shape[1]; k++) {
                sum += a_data[i * a_shape[1] + k] * b_data[k * b_shape[1] + j];
            }
            c_data[i * b_shape[1] + j] = sum;
        }
    }

    return c;
}

auto matmul::apply() -> tensor_t {
    assert(inputs.size() == 2);
    auto a = dynamic_cast<const tensor_t*>(inputs[0]);
    auto b = dynamic_cast<const tensor_t*>(inputs[1]);
    return _matmul_(a, b);
}

auto matmul::gradient(tensor_t&& grad) -> vector<tensor_t> {
    //TODO(): use function you get to complete
    //TODO();
    auto a = dynamic_cast<const tensor_t*>(inputs[0]);
    auto b = dynamic_cast<const tensor_t*>(inputs[1]);
   
 
    auto b_transposed = transpose(*b);
    auto a_transposed = transpose(*a);

    auto grad_a = _matmul_(&grad, &b_transposed);
    auto grad_b = _matmul_(&a_transposed, &grad);
   
    return {std::move(grad_a), std::move(grad_b)};
}

auto matmul::get_shape() -> vector<size_t> {
    //TODO(): NULL
    //TODO();
    auto a_shape = inputs[0]->shape;
    auto b_shape = inputs[1]->shape;
    return {a_shape[0], b_shape[1]};
}