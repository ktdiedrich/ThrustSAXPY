/*=========================================================================
*
*  Copyright (c) 2017  Karl T. Diedrich, PhD
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/count.h>

#include <iostream>  
#include <list>
#include <vector>


/** SAXPY implementation 
SAXPY operation y <-a * x + y
x = vector, y = vector, a = scalar constant

Based on https://github.com/thrust/thrust/wiki/Quick-Start-Guide 


@author Karl Diedrich, PhD ktdiedrich@gmail.com
*/ 

/* Simplify names with aliases */
typedef float CalcNumber;
typedef thrust::device_vector<CalcNumber> DeviceNumVector;
int const TEST_SIZE = 10000000; // Number of iterations for test

// Function definitions, can be in header file. In one file for simplicity of sample project. 

/** Fast SAXPY operator with lambda call
Lambda implementation requires nvcc compile option: --expt-extended-lambda */
void saxpy_fast(CalcNumber A, DeviceNumVector & X, DeviceNumVector & Y);


/** SAXPY y <-a * x + y in steps  */
void saxpy_slow(CalcNumber A, DeviceNumVector & X, DeviceNumVector & Y);

/** Test example SAXPY operation y <- a * x + y
x = vector, y = vector, a = scalar constant
*/
void testSaxpy();

/** */
void testReduce();

int main()
{
	testSaxpy();
	testReduce();
    return 0;
}


// Function implementations 
void saxpy_fast(CalcNumber A, DeviceNumVector & X, DeviceNumVector & Y)
{
	// auto saxpy_lambda = [=] __device__(auto x, auto y) { return A * x + y; };
	auto saxpy_lambda = [=] __device__(CalcNumber x, CalcNumber y) { return A * x + y; };
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_lambda);
}

void saxpy_slow(CalcNumber A, DeviceNumVector & X, DeviceNumVector & Y)
{
	DeviceNumVector temp(X.size());
	// temp <- A 
	thrust::fill(temp.begin(), temp.end(), A);
	// temp <- A * x
	thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<CalcNumber>());
	// Y <- A * X + Y
	thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<CalcNumber>());
}


void testSaxpy()
{
	// CalcNumber expected[] = { 3,  5,  7,  9, 11, 13, 15, 17, 19, 21 };
	CalcNumber const a1 = 2.0;
	CalcNumber const addY = 4.0;
	int const VECTOR_SIZE = TEST_SIZE;
	DeviceNumVector x1(VECTOR_SIZE);
	DeviceNumVector y1(VECTOR_SIZE);
	thrust::sequence(x1.begin(), x1.end());
	thrust::fill(y1.begin(), y1.end(), addY);
	std::cout << "SAXPY test: Y <- A*X+Y\na1 <-" << a1 << std::endl;
	std::cout << "x1 <- ";
	thrust::copy(x1.begin(), x1.end(), std::ostream_iterator<CalcNumber>(std::cout, ", "));
	std::cout << std::endl;

	std::cout << "y1 <- ";
	thrust::copy(y1.begin(), y1.end(), std::ostream_iterator<CalcNumber>(std::cout, ", "));
	std::cout << std::endl;
	saxpy_fast(a1, x1, y1);
	// saxpy_slow(a1, x1, y1);
	std::cout << "y1 transformed: ";
	thrust::copy(y1.begin(), y1.end(), std::ostream_iterator<CalcNumber>(std::cout, ", "));
	std::cout << std::endl;
}

void testReduce()
{
	DeviceNumVector D(TEST_SIZE);
	thrust::sequence(D.begin(), D.end());
	std::cout << "Reduce: ";
	thrust::copy(D.begin(), D.end(), std::ostream_iterator<CalcNumber>(std::cout, ", "));
	auto calcSum = thrust::reduce(D.begin(), D.end(), static_cast<CalcNumber>(0), thrust::plus<CalcNumber>());
	std::cout << "sum = " << calcSum << std::endl;

	thrust::device_vector<int> vec(6, 0);
	vec[0] = 1; vec[1] = 2; vec[2] = 2;  vec[3] = 3; vec[4] = 1;
	std::cout << "Vector: ";
	thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, ", "));
	int ones = thrust::count(vec.begin(), vec.end(), 1);
	std::cout << "Ones = " << ones << std::endl;
}
