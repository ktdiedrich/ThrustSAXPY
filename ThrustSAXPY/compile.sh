#!/bin/bash

#=========================================================================
#
#  Copyright (c) 2017  Karl T. Diedrich, PhD
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================*/

# Linux compile command, may need to update g++ version

COMPUTE_CAP=89
mkdir -p ../bin
nvcc --expt-extended-lambda -ccbin g++ -std c++11 -gencode=arch=compute_${COMPUTE_CAP},code=\"sm_${COMPUTE_CAP},compute_${COMPUTE_CAP}\" kernel.cu -o ../bin/ThrustSAXPY
