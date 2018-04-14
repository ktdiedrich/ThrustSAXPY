# ThrustSAXPY

SAXPY operation: y &lt;-a * x + y x = vector, y = vector, a = scalar constant ; on GPU

@author Karl Diedrich, PhD ktdiedrich@gmail.com

## Formula

SAXPY operation y <-a * x + y
x = vector, y = vector, a = scalar constant

## Implementation

Based on https://github.com/thrust/thrust/wiki/Quick-Start-Guide

## Project

Started with Visual Studio CUDA template

## Compile

### Windows compile

VS 2017 compiler not always compatible with CUDA toolkit
Project -> PROJECT_NAME properties -> Configuration Properties-> general -> Platform toolset: choose Visual Studio 2015 (v140)
Project-> PROJECT_NAME properties -> Configuration Properties -> CUDA C/C++ -> Command Line -> Additional Options: --expt-extended-lambda  

Command line compile:
1>...\ThrustSAXPY\ThrustSAXPY>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64" -x cu  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include"  -G   --keep-dir x64\Debug -maxrregcount=0  --machine 64 --compile  --expt-extended-lambda  -g   -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MDd " -o x64\Debug\kernel.cu.obj "...\ThrustSAXPY\ThrustSAXPY\kernel.cu" -clean
1>kernel.cu
1>Compiling CUDA source file kernel.cu...
1>
1>...\ThrustSAXPY\ThrustSAXPY>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin\nvcc.exe" -gencode=arch=compute_30,code=\"sm_30,compute_30\" --use-local-env --cl-version 2015 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64" -x cu  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include"  -G   --keep-dir x64\Debug -maxrregcount=0  --machine 64 --compile -cudart static --expt-extended-lambda -g   -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MDd " -o x64\Debug\kernel.cu.obj "...\ThrustSAXPY\ThrustSAXPY\kernel.cu"


### Linux compile

see ThrustSAXPY/compile.sh script

## License

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
