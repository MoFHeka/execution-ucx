/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#ifndef AXON_PYTHON_BINDINGS_ENUMS_HPP_
#define AXON_PYTHON_BINDINGS_ENUMS_HPP_

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void RegisterEnums(nb::module_& m);

#endif  // AXON_PYTHON_BINDINGS_ENUMS_HPP_
