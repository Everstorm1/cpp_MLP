# cpp_MLP
This is a small project where I will simply build a MLP AI model in pure C++ with no Tensor libraries, just math and std namespace


Currently I have a functional forward pass implemented and a way to save and load a model (plain text for readability, ofc not suitable for larger models where optimization is the focus)
Weights and biases are randomly generated, model structure can be defined with simple syntax (calling a function and just passing all layers as args)

----------------
Backpropagation, CCE, optimizer (Adam), etc. have yet to be implemented.
