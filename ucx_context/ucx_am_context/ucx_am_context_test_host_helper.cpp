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

#include "ucx_context/ucx_am_context/ucx_am_context_test_helper.h"

void processRecvDataHost(ucx_am_data_t& recvData) {
  float* data = static_cast<float*>(recvData.data);
  size_t size = recvData.data_length / sizeof(float);
  for (size_t i = 0; i < size; i++) {
    data[i] /= 2;
  }
}
