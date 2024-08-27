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
   
    for (size_t i = 0; i < Y.data.size(); ++i) {
        grad.data[i] = Y.data[i] - Y_hat[i];
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

