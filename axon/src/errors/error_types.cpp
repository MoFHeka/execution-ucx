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

#include "axon/errors/error_types.hpp"

#include <exception>
#include <string>
#include <system_error>
#include <utility>

namespace eux {
namespace axon {
namespace errors {

namespace {

class AxonErrorCategoryImpl : public std::error_category {
 public:
  const char* name() const noexcept override { return "AxonErrc"; }
  std::string message(int ev) const override {
    switch (static_cast<AxonErrc>(ev)) {
      case AxonErrc::Ok:
        return "OK";
      case AxonErrc::CoordinatorError:
        return "Coordinator error";
      case AxonErrc::ConnectError:
        return "Connect error";
      case AxonErrc::SerializeError:
        return "Serialize error";
      case AxonErrc::StorageBackpressure:
        return "Storage backpressure";
      case AxonErrc::DeserializeError:
        return "Deserialize error";
      default:
        return "Unknown error";
    }
  }
};

// Helper struct to register the category at static initialization time.
struct AxonErrorCategoryRegistrar {
  AxonErrorCategoryRegistrar() {
    rpc::RpcCategoryRegistry::GetInstance().RegisterCategory(
      AxonErrcCategory());
  }
};
static AxonErrorCategoryRegistrar axon_error_category_registrar_;

}  // namespace

const std::error_category& AxonErrcCategory() {
  static AxonErrorCategoryImpl instance;
  return instance;
}

std::error_code make_error_code(AxonErrc e) {
  return {static_cast<int>(e), AxonErrcCategory()};
}

std::string ToString(AxonErrc e) { return make_error_code(e).message(); }

std::exception_ptr MakeExceptionPtr(AxonErrorContext& context) {
  return std::make_exception_ptr(AxonErrorException(context));
}

std::exception_ptr MakeExceptionPtr(const AxonErrorContext& context) {
  return std::make_exception_ptr(AxonErrorException(context));
}

std::exception_ptr MakeExceptionPtr(AxonErrorContext&& context) {
  return std::make_exception_ptr(AxonErrorException(std::move(context)));
}

}  // namespace errors
}  // namespace axon
}  // namespace eux
