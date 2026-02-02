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

#ifndef AXON_PYTHON_PYTHON_MODULE_HPP_
#define AXON_PYTHON_PYTHON_MODULE_HPP_

#include <nanobind/nanobind.h>

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;

// Get cached asyncio module
nb::module_ GetAsyncioModule();

// Get cached builtins module
nb::module_ GetBuiltinsModule();

// Get cached inspect module
nb::module_ GetInspectModule();

// Get cached typing module
nb::module_ GetTypingModule();

// Get cached weakref module
nb::module_ GetWeakRefModule();

}  // namespace python
}  // namespace axon
}  // namespace eux

#endif  // AXON_PYTHON_PYTHON_MODULE_HPP_
