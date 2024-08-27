#include "easytf.h"
#include "optimizer.h"
#include "tensor.h"

/*
optimizer_t::optimizer_t(tensor_t* input, tensor_t* output) {
    TODO();
}

optimizer_t::optimizer_t(tensor_t* input, tensor_t* output, float_t lr) {
    TODO();
}*/

optimizer_t::optimizer_t(tensor_t* input, tensor_t* output) : input(input), root(output) {}

optimizer_t::optimizer_t(tensor_t* input, tensor_t* output, float_t lr) : input(input), root(output), lr(lr) {}
auto optimizer_t::get_grads(tensor_t&& Y) -> tensor_t {
   /* auto& Y_hat = root->data;

    //TODO(): init grad(tensor_t)
    TODO();

    //TODO(): get the per grad from the abs(y_hat - y)
    TODO();*/
    auto& Y_hat = root->data;
    auto grad = tensor_t(root->shape);
   
    for (size_t i = 0; i < Y_hat.size(); ++i) {
        grad.data[i] = Y_hat[i] - Y.data[i];
    }
   
    return grad;
}

void optimizer_t::step(tensor_t&& X, tensor_t&& Y) {
    //TODO(): NULL
    //TODO();
    input->set_data(X.data);
    root->forward();
    auto grad = get_grads(std::move(Y));
    root->backward(std::move(grad), lr);
}

#include "operator.h"
#include <cmath>
#include "easytf.h"
#include "node.h"
#include "tensor.h"

/*
operator_t::operator_t(const vector<node_t*>& inputs) {
   // TODO(): It seems that the inputs not be initialized
   TODO();
}*/

operator_t::operator_t(const vector<node_t*>& inputs) {
    this->inputs=inputs;
}

auto operator_t::forward() -> void {
    // TODO(): class operator_t solve the node_t is different from the tensor_t
    // think the key difference, you will get the tensorflow the core of calculated photo
    //TODO();
    //node_t::forward();
    for (auto& input : inputs) {
        input->forward();
    }
    auto result = apply();
    data = result.data;
    shape = result.shape;
}

auto operator_t::backward(tensor_t&& grad, float_t lr) -> void { // tensor_t&& grad is the right value reference
    // TODO(): base on the basic class, you will get more ideas
    //TODO();
    auto grads = gradient(std::move(grad));
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i]->backward(std::move(grads[i]), lr);
    }
};