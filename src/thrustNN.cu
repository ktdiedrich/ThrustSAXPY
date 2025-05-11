#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <iostream>
#include <cmath>
#include <vector>

// Activation function (Sigmoid)
struct sigmoid
{
    __host__ __device__ float operator()(float x) const
    {
        return 1.0f / (1.0f + expf(-x));
    }
};

// Derivative of the sigmoid function
struct sigmoid_derivative
{
    __host__ __device__ float operator()(float x) const
    {
        return x * (1.0f - x);
    }
};

// Multiply two vectors element-wise
struct multiply
{
    __host__ __device__ float operator()(float x, float y) const
    {
        return x * y;
    }
};

thrust::host_vector<float> initialize_host_vector(std::initializer_list<float> values)
{
    thrust::host_vector<float> vec(values.size());
    std::copy(values.begin(), values.end(), vec.begin());
    return vec;
}

int main()
{
    // Input data (2 samples, 3 features each)
    thrust::host_vector<float> h_input = initialize_host_vector({0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f}); 
    thrust::device_vector<float> d_input = h_input;

    // Weights (3 input features -> 1 output neuron)
    thrust::host_vector<float> h_weights = initialize_host_vector({0.5f, -0.5f, 0.3f});
    thrust::device_vector<float> d_weights = h_weights;

    // Expected output (2 samples)
    thrust::host_vector<float> h_expected = initialize_host_vector({0.0f, 1.0f});
    thrust::device_vector<float> d_expected = h_expected;

    // Output of the neural network
    thrust::device_vector<float> d_output(2);

    // Learning rate
    float learning_rate = 0.1f;

    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch)
    {
        // Compute weighted sum (dot product of input and weights)
        for (int i = 0; i < 2; ++i) // Loop over samples
        {
            float weighted_sum = thrust::inner_product(
                d_input.begin() + i * 3, d_input.begin() + (i + 1) * 3,
                d_weights.begin(), 0.0f);

            // Apply activation function (sigmoid)
            d_output[i] = sigmoid()(weighted_sum);
        }

        // Compute error (expected - output)
        thrust::device_vector<float> d_error(2);
        thrust::transform(d_expected.begin(), d_expected.end(),
                          d_output.begin(), d_error.begin(),
                          thrust::minus<float>());

        // Compute gradient (error * sigmoid_derivative(output))
        thrust::device_vector<float> d_gradient(2);
        thrust::transform(d_error.begin(), d_error.end(),
                          d_output.begin(), d_gradient.begin(),
                          multiply());

        // Update weights
        for (int i = 0; i < 3; ++i) // Loop over weights
        {
            float weight_update = 0.0f;
            for (int j = 0; j < 2; ++j) // Loop over samples
            {
                weight_update += d_gradient[j] * d_input[j * 3 + i];
            }
            d_weights[i] += learning_rate * weight_update;
        }
    }

    // Print final weights
    std::cout << "Final weights: ";
    thrust::copy(d_weights.begin(), d_weights.end(),
                 std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}