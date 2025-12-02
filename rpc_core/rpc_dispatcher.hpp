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

#include "ucx_context/ucx_context_data.hpp"
#ifndef RPC_CORE_RPC_DISPATCHER_HPP_
#define RPC_CORE_RPC_DISPATCHER_HPP_

#include <proxy/proxy.h>

#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "rpc_core/rpc_facade_helpers.hpp"
#include "rpc_core/rpc_response_builder.hpp"
#include "rpc_core/rpc_traits.hpp"
#include "rpc_core/rpc_types.hpp"
#include "rpc_core/utils/cista_serialize_helper.hpp"

namespace eux {
namespace rpc {

class RpcDispatcher;

/**
 * @brief Holds the result of an RPC invocation.
 *
 * This struct contains the response header and an optional, type-erased context
 * object that may be returned by the invoked function.
 *
 * @tparam ContextT The type of the payload object. Defaults to std::monostate
 * if no payload is returned.
 */
template <typename ReturnedPayload>
struct RpcInvokeResult {
  RpcResponseHeader header;
  ReturnedPayload payload;
};

/**
 * @brief A non-owning smart pointer for passing a type-erased context.
 *
 * Used internally to pass an optional context to the RPC function wrapper
 * without transferring ownership.
 */
using RpcContextPtr = std::unique_ptr<void, void (*)(void*)>;

struct ErasedRpcFunctionFacade
  : pro::facade_builder  //
    ::add_convention<
      pro::operator_dispatch<"()">,
      std::pair<RpcResponseHeader, ReturnedPayload>(
        const RpcRequestHeader&, RpcContextPtr, const RpcResponseBuilder&)>  //
    ::support_relocation<pro::constraint_level::trivial>                     //
    ::build {};

using ErasedRpcFunction = pro::proxy<ErasedRpcFunctionFacade>;

// Define operation tag of pro::proxy Facade
using RpcCallOperator = pro::operator_dispatch<"()">;

// Define function signatures in pro::proxy Facade for dynamic RPC
using DynamicFunctionReturnType =
  std::pair<data::vector<ParamMeta>, ReturnedPayload>;

namespace detail {
template <typename PayloadT>
using DynamicRpcConventionSignature =
  DynamicFunctionReturnType(const data::vector<ParamMeta>&, const PayloadT&);
using DynamicRpcConventionSignatureNoPayload =
  DynamicFunctionReturnType(const data::vector<ParamMeta>&);

// Define initial builder (contains "no payload" base version)
using InitialBuilder = pro::facade_builder  //
  ::add_skill<pro::skills::as_view>         //
  ::add_convention<
    pro::weak_dispatch<RpcCallOperator>,
    DynamicRpcConventionSignatureNoPayload>;
// Use TMP helper template to add all Payload conventions
using BuilderWithPayloads = AddVariantConventionsType<
  InitialBuilder, PayloadVariant, DynamicRpcConventionSignature,
  RpcCallOperator>;
}  // namespace detail

// Final build Facade
struct DynamicRpcFunctionFacade
  : detail::BuilderWithPayloads  //
    ::template support_relocation<pro::constraint_level::trivial>::build {};

using DynamicRpcFunction = pro::proxy<DynamicRpcFunctionFacade>;
using DynamicRpcFunctionView = pro::proxy_view<DynamicRpcFunctionFacade>;

// Helper for extracting function arguments from the request header and
// context.
template <typename T, typename Context>
T extract_arg(
  const RpcRequestHeader& req, Context&& context,
  const RpcResponseBuilder& builder, size_t& param_idx) {
  using DecayedT = std::decay_t<T>;
  static constexpr bool is_arithmetic_or_enum =
    std::is_arithmetic_v<DecayedT> || std::is_enum_v<DecayedT>;
  static constexpr bool is_response_builder = std::is_same_v<
    DecayedT, std::remove_reference_t<std::remove_const_t<RpcResponseBuilder>>>;
  static constexpr bool is_void_pointer =
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, void*>;

  if constexpr (std::is_same_v<DecayedT, RpcRequestHeader>) {
    return req;
  } else if constexpr (is_response_builder) {
    static_assert(
      std::is_lvalue_reference_v<T>
        && std::is_const_v<std::remove_reference_t<T>>,
      "RpcResponseBuilder must be passed as const&");
    return builder;
  } else if constexpr (is_arithmetic_or_enum) {
    return req.GetPrimitive<DecayedT>(param_idx++);
  } else if constexpr (is_cista_strong<DecayedT>::value) {
    using UnderlyingType =
      typename get_cista_strong_underlying_type<DecayedT>::type;
    return DecayedT{req.GetPrimitive<UnderlyingType>(param_idx++)};
  } else if constexpr (std::is_same_v<DecayedT, data::string>) {
    return req.GetString(param_idx++);
  } else if constexpr (std::is_same_v<DecayedT, TensorMeta>) {
    static_assert(
      std::is_lvalue_reference_v<T>,
      "TensorMeta must be passed by lvalue reference (e.g., const "
      "TensorMeta&) since RpcRequestHeader is const.");
    return req.GetTensor(param_idx++);
  } else if constexpr (is_data_vector<DecayedT>::value) {
    static_assert(
      std::is_lvalue_reference_v<T>,
      "cista::vector must be passed by lvalue reference (e.g., const "
      "cista::vector<T>&) since RpcRequestHeader is const.");
    using ElementType = typename DecayedT::value_type;
    return req.template GetVector<ElementType>(param_idx++);
  } else {
    // It's a context type.
    static_assert(
      is_payload_v<DecayedT>,
      "Function argument is not a serializable type or a valid payload type.");
    static_assert(
      !std::is_void_v<std::decay_t<Context>>,
      "Function expects a context, but none was provided.");
    static_assert(
      !is_void_pointer,
      "Function argument is a void pointer, but it should be a context type.");
    if constexpr (std::is_lvalue_reference_v<T>) {
      return *context;
    } else {
      return std::move(*context);
    }
  }
}

template <typename T>
struct is_std_tuple : std::false_type {};
template <typename... Args>
struct is_std_tuple<std::tuple<Args...>> : std::true_type {};

template <typename ArgsTuple, typename Context, size_t... Is>
auto extract_args_impl(
  const RpcRequestHeader& req, Context&& context,
  const RpcResponseBuilder& builder, std::index_sequence<Is...>) {
  [[maybe_unused]] size_t param_idx = 0;
  return std::tuple<std::tuple_element_t<Is, ArgsTuple>...>{
    extract_arg<std::tuple_element_t<Is, ArgsTuple>>(
      req, std::forward<Context>(context), builder, param_idx)...};
}

namespace detail {

template <typename Signature>
struct NaturalCallerFacade;

template <typename R, typename... Args>
struct NaturalCallerFacade<R(Args...)>
  : pro::facade_builder::template add_convention<
      pro::operator_dispatch<"()">, R(Args...)>::
      template support_relocation<pro::constraint_level::nontrivial>::
        template support_copy<pro::constraint_level::nontrivial>::build {};

}  // namespace detail

template size_t utils::GetSerializedSize(const RpcResponseHeader& value);
template size_t utils::GetSerializedSize(const RpcRequestHeader& value);

/**
 * @brief Manages registration and dispatching of RPC functions.
 *
 * The RpcDispatcher provides a type-safe mechanism to register C++ functions
 * and invoke them remotely. It handles serialization/deserialization of
 * arguments and return values, supports optional context passing, and provides
 * service discovery through function signatures.
 */
class RpcDispatcher {
 public:
  /**
   * @brief Constructs an RpcDispatcher with a unique instance name.
   *
   * @param instance_name A name to identify this dispatcher instance, used in
   * function signatures for service discovery.
   */
  explicit RpcDispatcher(data::string instance_name)
    : instance_name_(std::move(instance_name)) {}

  std::pair<RpcResponseHeader, ReturnedPayload> InvokeFunction(
    const RpcRequestHeader& request, RpcContextPtr context,
    const RpcResponseBuilder& builder) {
    try {
      auto it = functions_.find(request.function_id);
      if (it == functions_.end()) {
        throw std::runtime_error("Function not found");
      }
      return (*it->second)(request, std::move(context), builder);
    } catch (const std::exception& e) {
      RpcResponseHeader error_response;
      error_response.session_id = request.session_id;
      error_response.request_id = request.request_id;
      error_response.status =
        std::make_error_code(rpc::RpcErrc::INVALID_ARGUMENT);
      return {std::move(error_response), std::monostate{}};
    }
  }

  /**
   * @brief Serializes an RpcResponseHeader into a byte buffer.
   * @param response_header The response header to serialize.
   * @return A cista::byte_buf containing the serialized data.
   */
  static cista::byte_buf SerializeResponse(
    const RpcResponseHeader& response_header) {
    return cista::serialize<utils::SerializerMode>(response_header);
  }

  template <typename T>
  void SerializeResponse(const RpcResponseHeader& response_header, T& writer) {
    static_assert(
      std::is_same_v<
        decltype(std::declval<T&>().write(
          static_cast<void const*>(nullptr),
          std::declval<std::size_t>(),
          std::declval<std::size_t>())),
        cista::offset_t>,
      "Writer must implement: offset_t write(void const* ptr, std::size_t "
      "num_bytes, std::size_t alignment = 0U);");
    return cista::serialize<utils::SerializerMode>(writer, response_header);
  }

  static void SerializeResponse(
    const RpcResponseHeader& response_header,
    utils::FixedBufferWriter& writer) {
    return cista::serialize<utils::SerializerMode>(writer, response_header);
  }

  static void SerializeResponse(
    const RpcResponseHeader& response_header,
    cista::buf<cista::byte_buf>& buffer) {
    return cista::serialize<utils::SerializerMode>(buffer, response_header);
  }

  static ResponseHeaderUniquePtr DeerializeResponse(ucxx::UcxHeader&& data) {
    const auto* header =
      cista::deserialize<const RpcResponseHeader, utils::SerializerMode>(
        std::string_view(
          reinterpret_cast<const char*>(data.data()), data.size()));
    return ResponseHeaderUniquePtr(header, UcxHeaderDeleter{std::move(data)});
  }

  /**
   * @brief Registers a callable as an RPC function.
   *
   * @tparam Func The type of the callable (e.g., function pointer, lambda).
   * @param id The unique identifier for this function.
   * @param func The callable to register.
   * @param name A human-readable name for the function, used for service
   * discovery. Defaults to "anonymous".
   */
  template <typename Func>
  void RegisterFunction(
    function_id_t id, Func&& func,
    const data::string& name = data::string{"anonymous"}) {
    functions_.emplace(id, MakeRpcWrapper(std::forward<Func>(func)));

#ifdef EUX_RPC_ENABLE_NATURAL_CALL
    using Signature = typename signature_from_traits<std::decay_t<Func>>::type;
    native_functions_.emplace(
      id,
      pro::make_proxy<detail::NaturalCallerFacade<Signature>>(
        std::forward<Func>(func)));
#endif

    signatures_[id] = MakeRpcSignature<Func>(id, name, instance_name_);
  }

  /**
   * @brief Registers a type-erased dynamic function as an RPC function.
   *
   * This overload is designed for dynamic registration scenarios, such as
   * plugins or language bindings (e.g., Python), where function signatures are
   * not known at compile time.
   *
   * @param id The unique identifier for this function.
   * @param name A human-readable name for the function.
   * @param param_types A vector describing the types of serializable
   * parameters.
   * @param return_types A vector describing the types of serializable return
   * values.
   * @param input_payload_type The type of the non-serializable payload
   * expected as input, if any.
   * @param return_payload_type The type of the non-serializable payload
   * returned, if any.
   * @param func A type-erased callable that takes a vector of `ParamMeta` and
   * an `InputPayloadVariant` and returns a pair of a vector of result
   * `ParamMeta` and a `ReturnedPayload`.
   */
  void RegisterFunction(
    function_id_t id, const data::string& name,
    const data::vector<ParamType>& param_types,
    const data::vector<ParamType>& return_types, PayloadType input_payload_type,
    PayloadType return_payload_type, DynamicRpcFunction&& func) {
    functions_.emplace(
      id, pro::make_proxy<ErasedRpcFunctionFacade>(
            DynamicFuncWrapper<DynamicRpcFunction>{
              std::move(func), input_payload_type}));

    // Create and store the signature manually from the provided types.
    signatures_.emplace(
      id, RpcFunctionSignature{
            .instance_name = instance_name_,
            .id = id,
            .function_name = name,
            .param_types = param_types,
            .return_types = return_types,
            .input_payload_type = input_payload_type,
            .return_payload_type = return_payload_type,
          });
  }

  /**
   * @brief Deserializes a request, invokes the corresponding function with a
   * context passed by reference, and returns the result.
   *
   * @tparam ReturnContextT The expected type of the context returned by the
   * function.
   * @tparam InputData The type of the context data to be passed to the
   * function.
   * @param request_buffer The serialized request buffer.
   * @param input_data A reference to the context data.
   * @return An RpcInvokeResult containing the response header and any returned
   * context.
   */
  template <typename ReturnContextT, typename InputData>
  RpcInvokeResult<ReturnContextT> Dispatch(
    cista::byte_buf&& request_buffer, InputData& input_data) {
    auto [header, context_variant] = DispatchImpl(
      std::move(request_buffer),
      RpcContextPtr(&input_data, [](void*) { /* non-owning */ }));

    return BuildRpcResult<ReturnContextT>(
      std::move(header), std::move(context_variant));
  }

  template <typename ReturnContextT, typename InputData>
  RpcInvokeResult<ReturnContextT> Dispatch(
    const RpcRequestHeader& request_header, InputData& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    auto [header, context_variant] = InvokeFunction(
      request_header, RpcContextPtr(&input_data, [](void*) {}), builder);

    return BuildRpcResult<ReturnContextT>(
      std::move(header), std::move(context_variant));
  }

  /**
   * @brief Deserializes a request, invokes the corresponding function by moving
   * a context into it, and returns the result.
   *
   * This is useful for contexts that manage resources and should be owned by
   * the callee, such as UcxBufferVec.
   *
   * @tparam ReturnContextT The expected type of the context returned by the
   * function.
   * @tparam InputData The type of the context data to be moved.
   * @param request_buffer The serialized request buffer.
   * @param input_data An rvalue reference to the context data.
   * @return An RpcInvokeResult containing the response header and any returned
   * context.
   */
  template <typename ReturnContextT, typename InputData>
  RpcInvokeResult<ReturnContextT> DispatchMove(
    cista::byte_buf&& request_buffer, InputData&& input_data) {
    // Take ownership of the input data by moving it to a local variable on the
    // stack. This avoids heap allocation, which is critical for RAII types that
    // manage their own special resources (e.g., UcxBuffer).
    using DecayedInput = std::decay_t<InputData>;
    DecayedInput owned_input_data(std::forward<InputData>(input_data));

    // Pass a non-owning pointer to the stack-allocated object. This is safe
    // because the entire RPC dispatch is synchronous, so `owned_input_data`
    // will outlive the execution of the called function.
    // RpcContextPtr is a observation, it does not own the data
    RpcContextPtr non_owning_ptr(&owned_input_data, [](void*) {});

    auto [header, context_variant] =
      DispatchImpl(std::move(request_buffer), std::move(non_owning_ptr));

    return BuildRpcResult<ReturnContextT>(
      std::move(header), std::move(context_variant));
  }

  template <typename ReturnContextT, typename InputData>
  RpcInvokeResult<ReturnContextT> DispatchMove(
    const RpcRequestHeader& request_header, InputData&& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    using DecayedInput = std::decay_t<InputData>;
    DecayedInput owned_input_data(std::forward<InputData>(input_data));

    RpcContextPtr non_owning_ptr(&owned_input_data, [](void*) {});

    auto [header, context_variant] =
      InvokeFunction(request_header, std::move(non_owning_ptr), builder);

    return BuildRpcResult<ReturnContextT>(
      std::move(header), std::move(context_variant));
  }

  /**
   * @brief Deserializes a request, invokes a function that takes no context,
   * and returns the result.
   *
   * @tparam ReturnContextT The expected type of the context returned by the
   * function.
   * @param request_buffer The serialized request buffer.
   * @return An RpcInvokeResult containing the response header and any returned
   * context.
   */
  template <typename ReturnContextT>
  RpcInvokeResult<ReturnContextT> Dispatch(cista::byte_buf&& request_buffer) {
    auto [header, context_variant] = DispatchImpl(
      std::move(request_buffer), RpcContextPtr(nullptr, [](void*) {}));

    return BuildRpcResult<ReturnContextT>(
      std::move(header), std::move(context_variant));
  }

  template <typename ReturnContextT>
  RpcInvokeResult<ReturnContextT> Dispatch(
    const RpcRequestHeader& request_header,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    auto [header, context_variant] = InvokeFunction(
      request_header, RpcContextPtr(nullptr, [](void*) {}), builder);

    return BuildRpcResult<ReturnContextT>(
      std::move(header), std::move(context_variant));
  }

  /**
   * @brief Dynamically dispatches a request without knowing the return context
   * type at compile time.
   *
   * This is the dynamic counterpart to the templated `dispatch` method. It
   * returns the raw `std::variant` of possible context types, allowing the
   * caller to inspect the result at runtime using `std::visit` or
   * `std::holds_alternative`.
   *
   * @param request_buffer The serialized request buffer.
   * @return An `RpcInvokeResult` containing the header and an optional variant
   * of the returned context.
   */
  RpcInvokeResult<ReturnedPayload> Dispatch(cista::byte_buf&& request_buffer) {
    auto [header, context_variant] = DispatchImpl(
      std::move(request_buffer), RpcContextPtr(nullptr, [](void*) {}));
    return {std::move(header), std::move(context_variant)};
  }

  RpcInvokeResult<ReturnedPayload> Dispatch(
    const RpcRequestHeader& request_header,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    auto [header, context_variant] = InvokeFunction(
      request_header, RpcContextPtr(nullptr, [](void*) {}), builder);
    return {std::move(header), std::move(context_variant)};
  }

  template <typename InputData>
  RpcInvokeResult<ReturnedPayload> Dispatch(
    cista::byte_buf&& request_buffer, const InputData& input_data) {
    auto [header, context_variant] = DispatchImpl(
      std::move(request_buffer),
      RpcContextPtr(
        const_cast<InputData*>(&input_data), [](void*) { /* non-owning */ }));
    return {std::move(header), std::move(context_variant)};
  }

  template <typename InputData>
  RpcInvokeResult<ReturnedPayload> Dispatch(
    const RpcRequestHeader& request_header, const InputData& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    auto [header, context_variant] = InvokeFunction(
      request_header,
      RpcContextPtr(const_cast<InputData*>(&input_data), [](void*) {}),
      builder);
    return {std::move(header), std::move(context_variant)};
  }

  template <typename InputData>
  RpcInvokeResult<ReturnedPayload> DispatchMove(
    cista::byte_buf&& request_buffer, InputData&& input_data) {
    using DecayedInput = std::decay_t<InputData>;
    DecayedInput owned_input_data(std::forward<InputData>(input_data));
    RpcContextPtr non_owning_ptr(&owned_input_data, [](void*) {});
    auto [header, context_variant] =
      DispatchImpl(std::move(request_buffer), std::move(non_owning_ptr));
    return {std::move(header), std::move(context_variant)};
  }

  template <typename InputData>
  RpcInvokeResult<ReturnedPayload> DispatchMove(
    const RpcRequestHeader& request_header, InputData&& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    using DecayedInput = std::decay_t<InputData>;
    DecayedInput owned_input_data(std::forward<InputData>(input_data));
    RpcContextPtr non_owning_ptr(&owned_input_data, [](void*) {});
    auto [header, context_variant] =
      InvokeFunction(request_header, std::move(non_owning_ptr), builder);
    return {std::move(header), std::move(context_variant)};
  }

  template <typename Func>
  std::pair<RpcResponseHeader, ReturnedPayload> DispatchAdhoc(
    Func&& func, const RpcRequestHeader& header, RpcContextPtr context,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    auto wrapper = MakeRpcWrapper(std::forward<Func>(func));
    return (*wrapper)(header, std::move(context), builder);
  }

  std::pair<RpcResponseHeader, ReturnedPayload> DispatchAdhoc(
    DynamicRpcFunctionView func, PayloadType input_payload_type,
    const RpcRequestHeader& header, RpcContextPtr context,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    auto wrapper = pro::make_proxy<ErasedRpcFunctionFacade>(
      DynamicFuncWrapper<DynamicRpcFunctionView>{func, input_payload_type});
    return (*wrapper)(header, std::move(context), builder);
  }

  /**
   * @brief Checks if a function with the given ID is registered.
   * @param id The function ID to check.
   * @return True if the function is registered, false otherwise.
   */
  bool IsRegistered(function_id_t id) const {
    return functions_.find(id) != functions_.end();
  }

  /**
   * @brief Gets the total number of registered functions.
   * @return The number of functions.
   */
  size_t FunctionCount() const { return functions_.size(); }

  /**
   * @brief Retrieves the signature of a registered function.
   * @param id The ID of the function.
   * @return An std::optional<RpcFunctionSignature> containing the signature if
   * found, otherwise std::nullopt.
   */
  std::optional<RpcFunctionSignature> GetSignature(function_id_t id) const {
    auto it = signatures_.find(id);
    if (it != signatures_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  /**
   * @brief Retrieves all function signatures and serializes them into a byte
   * buffer.
   *
   * This is useful for service discovery, allowing a client to query a server's
   * capabilities.
   *
   * @return A cista::byte_buf containing the serialized vector of
   * RpcFunctionSignature.
   */
  cista::byte_buf GetAllSignatures() const {
    data::vector<RpcFunctionSignature> sig_vec;
    for (const auto& pair : signatures_) {
      sig_vec.push_back(pair.second);
    }
    return cista::serialize<utils::SerializerMode>(sig_vec);
  }

/**
 * @brief Gets a callable that can be used to invoke a registered function.
 *
 * This is a type-safe way to call a function that was previously registered
 * with `register_function`.
 *
 * // Call it like a normal function
 * auto add_func = dispatcher.get_caller<int(int, int)>(function_id_t{1});
 * int result = add_func(5, 10); // result will be 15
 */
#ifdef EUX_RPC_ENABLE_NATURAL_CALL
  template <typename Signature>
  pro::proxy<detail::NaturalCallerFacade<Signature>> GetCaller(
    function_id_t id) {
    if (!IsRegistered(id)) {
      throw std::runtime_error(
        "Function with ID " + std::to_string(id.v_) + " not registered.");
    }
    const auto& any_func = native_functions_.at(id);
    using ProxyType = pro::proxy<detail::NaturalCallerFacade<Signature>>;
    return std::any_cast<ProxyType>(any_func);
  }
#endif

 private:
  cista::raw::hash_map<function_id_t, ErasedRpcFunction> functions_;
#ifdef EUX_RPC_ENABLE_NATURAL_CALL
  cista::raw::hash_map<function_id_t, std::any> native_functions_;
#endif

 protected:
  data::string instance_name_;
  cista::raw::hash_map<function_id_t, RpcFunctionSignature> signatures_;

  template <typename Func, typename PayloadT>
  static constexpr bool is_dynamic_function_v = requires(Func&& func) {
    {
      (*std::forward<Func>(func))(
        std::declval<const data::vector<ParamMeta>&>(),
        std::declval<const PayloadT&>())
    };  // NOLINT(readability/braces)
  };

  template <typename Func>
  struct DynamicFuncWrapper {
    Func dynamic_func;
    PayloadType input_payload_type;

    std::pair<RpcResponseHeader, ReturnedPayload> operator()(
      const RpcRequestHeader& request, RpcContextPtr context_ptr,
      const RpcResponseBuilder& builder) {
      constexpr bool is_reference_wrapper =
        is_reference_wrapper_v<std::decay_t<Func>>;
      constexpr bool is_dynamic_function =
        is_dynamic_function_v<Func, ucxx::UcxBuffer>
        || is_dynamic_function_v<Func, ucxx::UcxBufferVec>;
      try {
        auto call_fn = [&]() {
          if constexpr (is_dynamic_function) {
            if (context_ptr) {
              if (input_payload_type == PayloadType::UCX_BUFFER) {
                return (*dynamic_func)(
                  request.params,
                  *static_cast<ucxx::UcxBuffer*>(context_ptr.get()));
              } else if (input_payload_type == PayloadType::UCX_BUFFER_VEC) {
                return (*dynamic_func)(
                  request.params,
                  *static_cast<ucxx::UcxBufferVec*>(context_ptr.get()));
              }
            }
            return (*dynamic_func)(request.params);
          } else if constexpr (is_reference_wrapper) {
            if (context_ptr) {
              if (input_payload_type == PayloadType::UCX_BUFFER) {
                return (*(dynamic_func.get()))(
                  request.params,
                  *static_cast<ucxx::UcxBuffer*>(context_ptr.get()));
              } else if (input_payload_type == PayloadType::UCX_BUFFER_VEC) {
                return (*(dynamic_func.get()))(
                  request.params,
                  *static_cast<ucxx::UcxBufferVec*>(context_ptr.get()));
              }
            }
            return (*(dynamic_func.get()))(request.params);
          } else {
            static_assert(false, "Invalid function type");
          }
        };
        auto [serializable_results, returned_payload] = call_fn();
        RpcResponseHeader response_header;
        response_header.session_id = request.session_id;
        response_header.request_id = request.request_id;
        response_header.status = std::make_error_code(rpc::RpcErrc::OK);
        response_header.results = std::move(serializable_results);
        return {std::move(response_header), std::move(returned_payload)};
      } catch (const std::exception& e) {
        RpcResponseHeader error_response;
        error_response.session_id = request.session_id;
        error_response.request_id = request.request_id;
        error_response.status = std::make_error_code(rpc::RpcErrc::INTERNAL);
        ParamMeta error_meta;
        error_meta.type = ParamType::STRING;
        error_meta.value.emplace<data::string>(e.what());
        error_response.results.emplace_back(std::move(error_meta));
        return {std::move(error_response), std::monostate{}};
      }
    }
  };

  std::pair<RpcResponseHeader, ReturnedPayload> DispatchImpl(
    cista::byte_buf&& request_buffer, RpcContextPtr context) {
    const RpcRequestHeader* request_ptr = nullptr;
    try {
      request_ptr =
        cista::deserialize<const RpcRequestHeader, utils::SerializerMode>(
          request_buffer);
      if (!request_ptr) {
        throw std::runtime_error("Failed to deserialize request");
      }

      RpcResponseBuilder builder;
      return InvokeFunction(*request_ptr, std::move(context), builder);
    } catch (const std::exception& e) {
      RpcResponseHeader error_response;
      if (request_ptr) {
        error_response.session_id = request_ptr->session_id;
        error_response.request_id = request_ptr->request_id;
      }
      error_response.status =
        std::make_error_code(rpc::RpcErrc::INVALID_ARGUMENT);
      return {std::move(error_response), std::monostate{}};
    }
  }

  template <typename ReturnContextT>
  RpcInvokeResult<ReturnContextT> BuildRpcResult(
    RpcResponseHeader&& header, ReturnedPayload&& context_variant) {
    static_assert(
      std::is_same_v<ReturnContextT, std::monostate>
        || std::is_same_v<ReturnContextT, ucxx::UcxBuffer>
        || std::is_same_v<ReturnContextT, ucxx::UcxBufferVec>,
      "ReturnContextT must be std::monostate, ucxx::UcxBuffer, or "
      "ucxx::UcxBufferVec");

    return {
      std::move(header), std::get<ReturnContextT>(std::move(context_variant))};
  }

  template <typename Tuple, typename ContextType, size_t Index = 0>
  static void FillParamTypesImpl(data::vector<ParamType>& params) {
    if constexpr (Index < std::tuple_size_v<Tuple>) {
      using T = std::tuple_element_t<Index, Tuple>;
      using DecayedT = std::decay_t<T>;
      if constexpr (get_param_type<DecayedT>() != ParamType::UNKNOWN) {
        params.push_back(get_param_type<T>());
      }
      FillParamTypesImpl<Tuple, ContextType, Index + 1>(params);
    }
  }

  template <typename Tuple, size_t Index = 0>
  static void FillReturnTypesImpl(data::vector<ParamType>& return_types) {
    if constexpr (Index < std::tuple_size_v<Tuple>) {
      using T = std::tuple_element_t<Index, Tuple>;
      using DecayedT = std::decay_t<T>;
      if constexpr (get_param_type<DecayedT>() != ParamType::UNKNOWN) {
        return_types.push_back(get_param_type<T>());
      }
      FillReturnTypesImpl<Tuple, Index + 1>(return_types);
    }
  }

  template <typename Func>
  static RpcFunctionSignature MakeRpcSignature(
    function_id_t id, const data::string& function_name,
    const data::string& instance_name) {
    using Traits = function_traits<std::decay_t<Func>>;
    using ReturnType = typename Traits::return_type;
    using ArgsTuple = typename Traits::args_tuple;
    using InputContextType = typename payload_finder<ArgsTuple>::type;

    RpcFunctionSignature sig;
    sig.instance_name = instance_name;
    sig.id = id;
    sig.function_name = function_name;
    sig.input_payload_type = get_payload_type<InputContextType>();

    FillParamTypesImpl<ArgsTuple, InputContextType>(sig.param_types);

    using DecayR = std::decay_t<ReturnType>;
    static constexpr bool is_response_pair =
      is_pair<DecayR>::value
      && std::is_same_v<
        std::decay_t<typename get_pair_first_type<DecayR>::type>,
        RpcResponseHeader>;
    if constexpr (std::is_void_v<DecayR>) {
      sig.return_payload_type = PayloadType::NO_PAYLOAD;
    } else if constexpr (std::is_same_v<DecayR, RpcResponseHeader>) {
      sig.return_payload_type = PayloadType::NO_PAYLOAD;
    } else if constexpr (is_response_pair) {
      using PayloadT = std::decay_t<typename DecayR::second_type>;
      if constexpr (is_payload_v<PayloadT>) {
        sig.return_payload_type = get_payload_type<PayloadT>();
      } else {
        sig.return_payload_type = PayloadType::NO_PAYLOAD;
      }
    } else if constexpr (is_std_tuple<DecayR>::value) {
      FillReturnTypesImpl<DecayR>(sig.return_types);
      using PayloadT = typename payload_finder<DecayR>::type;
      if constexpr (!std::is_void_v<PayloadT>) {
        sig.return_payload_type = get_payload_type<PayloadT>();
      } else {
        sig.return_payload_type = PayloadType::NO_PAYLOAD;
      }
    } else {
      if (is_payload_v<DecayR>) {
        sig.return_payload_type = get_payload_type<DecayR>();
      } else {
        if (get_param_type<DecayR>() != ParamType::UNKNOWN) {
          sig.return_types.push_back(get_param_type<DecayR>());
        }
        sig.return_payload_type = PayloadType::NO_PAYLOAD;
      }
    }
    return sig;
  }

  template <typename Func>
  static pro::proxy<ErasedRpcFunctionFacade> MakeRpcWrapper(Func&& func) {
    return pro::make_proxy<ErasedRpcFunctionFacade>(
      [func = std::forward<Func>(func)](
        const RpcRequestHeader& request, RpcContextPtr context_ptr,
        const RpcResponseBuilder& builder)
        -> std::pair<RpcResponseHeader, ReturnedPayload> {
        try {
          using Traits = function_traits<std::decay_t<Func>>;
          using ReturnType = typename Traits::return_type;
          using ArgsTuple = typename Traits::args_tuple;
          using ContextType = typename payload_finder<ArgsTuple>::type;

          auto invoke_and_package =
            [&]() -> std::pair<RpcResponseHeader, ReturnedPayload> {
            auto args = [&]() {
              if constexpr (std::is_void_v<ContextType>) {
                return extract_args_impl<ArgsTuple>(
                  request, static_cast<void*>(nullptr), builder,
                  std::make_index_sequence<Traits::arity>{});
              } else {
                auto* typed_context =
                  static_cast<ContextType*>(context_ptr.get());
                if (!typed_context) {
                  throw std::runtime_error(
                    "Context required but not provided.");
                }
                return extract_args_impl<ArgsTuple>(
                  request, typed_context, builder,
                  std::make_index_sequence<Traits::arity>{});
              }
            }();

            if constexpr (std::is_void_v<ReturnType>) {
              // Case: Function returns void
              std::apply(func, std::move(args));
              auto header =
                builder.PrepareResponse(request.session_id, request.request_id);
              return {std::move(header), std::monostate{}};
            } else {
              // Case: Function returns a value
              auto result = std::apply(func, std::move(args));
              using DecayR = std::decay_t<decltype(result)>;

              // Manual mode: User returns a pre-built response header.
              static constexpr bool is_header_pair =
                is_pair<DecayR>::value
                && std::is_same_v<
                  std::decay_t<typename get_pair_first_type<DecayR>::type>,
                  RpcResponseHeader>;

              if constexpr (std::is_same_v<DecayR, RpcResponseHeader>) {
                return {std::move(result), std::monostate{}};
              } else if constexpr (is_header_pair) {
                return {
                  std::move(result.first),
                  ReturnedPayload(std::move(result.second))};
              } else {
                // Automatic mode: Build response from the function's return
                // value.
                auto builder_result = [&]() {
                  if constexpr (is_std_tuple<DecayR>::value) {
                    return std::apply(
                      [&](auto&&... results) {
                        return builder.PrepareResponse(
                          request.session_id, request.request_id,
                          std::forward<decltype(results)>(results)...);
                      },
                      std::move(result));
                  } else {
                    return builder.PrepareResponse(
                      request.session_id, request.request_id,
                      std::move(result));
                  }
                }();

                if constexpr (std::is_same_v<
                                std::decay_t<decltype(builder_result)>,
                                RpcResponseHeader>) {
                  return {std::move(builder_result), std::monostate{}};
                } else {
                  return {
                    std::move(builder_result.first),
                    std::move(builder_result.second)};
                }
              }
            }
          };

          return invoke_and_package();
        } catch (const std::exception& e) {
          RpcResponseHeader error_response;
          error_response.session_id = request.session_id;
          error_response.request_id = request.request_id;
          error_response.status = std::make_error_code(rpc::RpcErrc::INTERNAL);
          // In a real application, you would log e.what() here.
          // For debugging, pack the error message into the results.
          ParamMeta error_meta;
          error_meta.type = ParamType::STRING;
          error_meta.value.emplace<data::string>(e.what());
          error_response.results.emplace_back(std::move(error_meta));
          return {std::move(error_response), std::monostate{}};
        }
      });
  }
};

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_DISPATCHER_HPP_
