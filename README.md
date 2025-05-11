# Algorithms implemented in Nvida Thrust library for Nvdia GPUs.

## ThrustSAXPY

SAXPY operation: y &lt;-a * x + y x = vector, y = vector, a = scalar constant ; on GPU

@author Karl Diedrich, PhD ktdiedrich@gmail.com

### Formula

SAXPY operation y <-a * x + y
x = vector, y = vector, a = scalar constant

### Implementation

Based on https://github.com/thrust/thrust/wiki/Quick-Start-Guide


## Compile

### Environment

May need to adjust CUDA_LIB_PATH, default="-L/lib/x86_64-linux-gnu"
and CUDA_INCLUDE_PATH, default="-I/usr/include" in the Makefile.

### Windows compile

[Install Windows Susbsytem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) to compile and use GPUs.

### Linux compile

`make`

## Montitoring usage

### Watch GPU usage

```
 watch -n2 nvidia-smi
```

# Debugging

Under DEB package manger Linux or WSL 
Install debuggers: 

```
sudo apt update
sudo apt install gdb cuda-gdb
```

Configure `cuda-gdb` launchers for VS Code in `.vscode/launch.json`


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
