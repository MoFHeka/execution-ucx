/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.
 *
 *Licensed under the Apache License Version 2.0 with LLVM Exceptions
 *(the "License"); you may not use this file except in compliance with
 *the License. You may obtain a copy of the License at
 *
 *    https://llvm.org/LICENSE.txt
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *==============================================================================*/

#pragma once

#ifndef RPC_CORE_RPC_FACADE_HELPERS_HPP_
#define RPC_CORE_RPC_FACADE_HELPERS_HPP_

#include <proxy/proxy.h>

#include <variant>

namespace eux {
namespace rpc {

/**
 * @brief A generic TMP helper template that applies a "signature template"
 * to all types in a variant and adds them to the Builder.
 *
 * @tparam Builder The initial facade_builder type.
 * @tparam Variant The std::variant that contains all types to be added.
 * @tparam SignatureTemplate A template alias (such as RpcConventionSignature)
 *        which takes a type and produces a function signature.
 * @tparam DispatchTag The operator passed to signature. e.g.
 * pro::operator_dispatch<"()">.
 */
template <
  typename Builder, typename Variant,
  template <typename> typename SignatureTemplate, typename DispatchOperator>
struct AddVariantConventions;

// Specialization: Unpack std::variant
template <
  typename Builder, typename... Payloads,
  template <typename> typename SignatureTemplate, typename DispatchOperator>
struct AddVariantConventions<
  Builder, std::variant<Payloads...>, SignatureTemplate, DispatchOperator> {
 private:
  // Recursive folder
  template <typename CurrentBuilder, typename... Rest>
  struct Fold;

  // Base case
  template <typename CurrentBuilder>
  struct Fold<CurrentBuilder> {
    using type = CurrentBuilder;
  };

  // Recursive step
  template <
    typename CurrentBuilder, typename FirstPayload, typename... RestPayloads>
  struct Fold<CurrentBuilder, FirstPayload, RestPayloads...> {
    using NextBuilder = typename CurrentBuilder::template add_convention<
      pro::weak_dispatch<DispatchOperator>,
      SignatureTemplate<FirstPayload>  // <-- Apply signature template
      >;

    using type = typename Fold<NextBuilder, RestPayloads...>::type;
  };

 public:
  // Start recursive folding
  using type = typename Fold<Builder, Payloads...>::type;
};

/**
 * @brief Helper alias to make AddVariantConventions easier to use.
 */
template <
  typename Builder, typename Variant,
  template <typename> typename SignatureTemplate, typename DispatchOperator>
using AddVariantConventionsType = typename AddVariantConventions<
  Builder, Variant, SignatureTemplate, DispatchOperator>::type;

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_FACADE_HELPERS_HPP_
