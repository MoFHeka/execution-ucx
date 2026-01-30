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

#include <nanobind/nanobind.h>

#include "axon/python/bindings_enums.hpp"
#include "axon/python/bindings_runtime.hpp"
#include "axon/python/bindings_types.hpp"

namespace nb = nanobind;

// Forward declarations for bindings
void RegisterEnums(nb::module_& m);
void RegisterTypes(nb::module_& m);
void RegisterRuntime(nb::module_& m);

NB_MODULE(axon, m) {
  m.doc() = "Axon Runtime Python bindings";

  // Register all bindings
  RegisterEnums(m);
  RegisterTypes(m);
  RegisterRuntime(m);
}
