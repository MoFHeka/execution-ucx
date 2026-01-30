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

#include "axon/python/python_module.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;

//
// Cache Python modules to avoid repeated import_ calls
//

nb::module_ GetAsyncioModule() {
  static auto* asyncio = new nb::module_(nb::module_::import_("asyncio"));
  return *asyncio;
}

nb::module_ GetBuiltinsModule() {
  static auto* builtins = new nb::module_(nb::module_::import_("builtins"));
  return *builtins;
}

nb::module_ GetInspectModule() {
  static auto* inspect = new nb::module_(nb::module_::import_("inspect"));
  return *inspect;
}

nb::module_ GetTypingModule() {
  static auto* typing = new nb::module_(nb::module_::import_("typing"));
  return *typing;
}

nb::module_ GetWeakRefModule() {
  static auto* weakref = new nb::module_(nb::module_::import_("weakref"));
  return *weakref;
}

}  // namespace python
}  // namespace axon
}  // namespace eux
