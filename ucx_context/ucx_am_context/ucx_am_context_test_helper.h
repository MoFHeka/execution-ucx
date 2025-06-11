/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License Version 2.0 with LLVM Exceptions
(the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    https://llvm.org/LICENSE.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef UCX_CONTEXT_UCX_AM_CONTEXT_UCX_AM_CONTEXT_TEST_HELPER_H_
#define UCX_CONTEXT_UCX_AM_CONTEXT_UCX_AM_CONTEXT_TEST_HELPER_H_

#include "ucx_context/ucx_context_def.h"

void processRecvDataHost(ucx_am_data_t& recvData);

#if CUDA_ENABLED
void processRecvDataCuda(ucx_am_data_t& recvData);
#endif

#endif  // UCX_CONTEXT_UCX_AM_CONTEXT_UCX_AM_CONTEXT_TEST_HELPER_H_
