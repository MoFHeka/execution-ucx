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

#ifndef RPC_CORE_RPC_DISPATCHER_HPP_
#define RPC_CORE_RPC_DISPATCHER_HPP_

#include <any>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "rpc_core/rpc_types.hpp"

namespace stdexe_ucx_runtime {
namespace rpc_core {

#if WITH_CISTA_VERSION && WITH_CISTA_INTEGRITY
constexpr auto const MODE =  // opt. versioning + check sum
  cista::mode::WITH_VERSION | cista::mode::WITH_INTEGRITY;
#elif WITH_CISTA_INTEGRITY
constexpr auto const MODE = cista::mode::WITH_INTEGRITY;
#elif WITH_CISTA_VERSION
constexpr auto const MODE = cista::mode::WITH_VERSION;
#else
constexpr auto const MODE = cista::mode::NONE;
#endif

/**
 * @brief Holds the result of an RPC invocation.
 *
 * This struct contains the response header and an optional, type-erased context
 * object that may be returned by the invoked function.
 *
 * @tparam ContextT The type of the context object. Defaults to std::monostate
 * if no context is returned.
 */
template <typename ContextT = std::monostate>
struct RpcInvokeResult {
  RpcResponseHeader header;
  ContextT context;
};

/**
 * @brief A non-owning smart pointer for passing a type-erased context.
 *
 * Used internally to pass an optional context to the RPC function wrapper
 * without transferring ownership.
 */
using RpcContextPtr = std::unique_ptr<void, void (*)(void*)>;

/**
 * @brief A type-erased wrapper for a registered RPC function.
 *
 * It takes a request header and an optional context pointer and returns a pair
 * containing the response header and a type-erased context (if any).
 */
using ErasedRpcFunction = std::function<std::pair<RpcResponseHeader, std::any>(
  RpcRequestHeader&, RpcContextPtr)>;

// =============================================================================
// SFINAE Type Traits for C++17 Compatibility
// =============================================================================

// Checks if a type is a std::pair
template <typename T>
struct is_pair : std::false_type {};
template <typename T1, typename T2>
struct is_pair<std::pair<T1, T2>> : std::true_type {};

// Checks if a type is RpcResponseHeader
template <typename T>
struct is_rpc_response_header : std::false_type {};
template <>
struct is_rpc_response_header<RpcResponseHeader> : std::true_type {};

// Checks if a type is cista::strong
template <typename T>
struct is_cista_strong : std::false_type {};
template <typename T, typename Tag>
struct is_cista_strong<cista::strong<T, Tag>> : std::true_type {};

// Helper to get the underlying type of a cista::strong
template <typename T>
struct get_cista_strong_underlying_type;

template <typename T, typename Tag>
struct get_cista_strong_underlying_type<cista::strong<T, Tag>> {
  using type = T;
};

// Helper to safely get the first type of a pair
template <typename T>
struct get_pair_first_type {
  using type = void;
};
template <typename T1, typename T2>
struct get_pair_first_type<std::pair<T1, T2>> {
  using type = T1;
};

// C++20 function traits for deducing properties of callables.
template <typename T>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
  using return_type = R;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...)> {
  using return_type = R;
  using class_type = C;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const> {
  using return_type = R;
  using class_type = C;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename Lambda>
struct function_traits : function_traits<decltype(&Lambda::operator())> {};

// Helper to deduce the input context type from the function signature.
template <typename T>
struct is_data_vector : std::false_type {};
template <typename T>
struct is_data_vector<data::vector<T>> : std::true_type {};

template <typename Tuple, size_t Index = 0>
struct context_finder;

template <typename Tuple, size_t Index>
struct context_finder {
  using current_element_raw = std::tuple_element_t<Index, Tuple>;
  using current_element = std::decay_t<current_element_raw>;

  using type = std::conditional_t<
    !std::is_arithmetic_v<current_element> && !std::is_enum_v<current_element>
      && !is_cista_strong<current_element>::value
      && !std::is_same_v<current_element, RpcRequestHeader>
      && !is_data_vector<current_element>::value
      && !std::is_same_v<current_element, data::string>,
    current_element, typename context_finder<Tuple, Index + 1>::type>;
};

template <typename Tuple>
struct context_finder<Tuple, std::tuple_size_v<Tuple>> {
  using type = void;
};

// Helper for extracting function arguments from the request header and
// context.
template <typename T, typename Context>
T extract_arg(RpcRequestHeader& req, Context&& context, size_t& param_idx) {
  using DecayedT = std::decay_t<T>;
  static constexpr bool is_arithmetic_or_enum =
    std::is_arithmetic_v<DecayedT> || std::is_enum_v<DecayedT>;

  if constexpr (std::is_same_v<DecayedT, RpcRequestHeader>) {
    return std::is_rvalue_reference_v<T&&> ? std::move(req)
                                           : static_cast<T>(req);
  } else if constexpr (is_arithmetic_or_enum) {
    return req.get_primitive<DecayedT>(param_idx++);
  } else if constexpr (is_cista_strong<DecayedT>::value) {
    using UnderlyingType =
      typename get_cista_strong_underlying_type<DecayedT>::type;
    return DecayedT{req.get_primitive<UnderlyingType>(param_idx++)};
  } else if constexpr (std::is_same_v<DecayedT, data::string>) {
    return req.get_string(param_idx++);
  } else if constexpr (is_data_vector<DecayedT>::value) {
    using ElementType = typename DecayedT::value_type;
    // The ternary operator was causing a -Wreturn-local-addr warning because
    // the two branches have different return types (value vs. reference),
    // forcing the reference to be materialized into a temporary.
    // Using if constexpr solves this by compiling only one branch.
    if constexpr (!std::is_lvalue_reference_v<T>) {
      // For value and rvalue reference types, move the vector.
      return req.template move_vector<ElementType>(param_idx++);
    } else {
      // For lvalue reference types, return a reference to avoid copies.
      return req.template get_vector<ElementType>(param_idx++);
    }
  } else {
    // It's a context type.
    static_assert(
      !std::is_void_v<std::decay_t<Context>>,
      "Function expects a context, but none was provided.");
    if constexpr (std::is_lvalue_reference_v<T>) {
      return *context;
    } else {
      return std::move(*context);
    }
  }
}

template <typename ArgsTuple, typename Context, size_t... Is>
auto extract_args_impl(
  RpcRequestHeader& req, Context&& context, std::index_sequence<Is...>) {
  [[maybe_unused]] size_t param_idx = 0;
  return std::tuple<std::tuple_element_t<Is, ArgsTuple>...>{
    extract_arg<std::tuple_element_t<Is, ArgsTuple>>(
      req, std::forward<Context>(context), param_idx)...};
}

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

 private:
  data::string instance_name_;
  cista::offset::hash_map<function_id_t, ErasedRpcFunction> functions_;
  cista::offset::hash_map<function_id_t, RpcFunctionSignature> signatures_;

  std::pair<RpcResponseHeader, std::any> dispatch_impl(
    cista::byte_buf&& request_buffer, RpcContextPtr context) {
    RpcRequestHeader* request_ptr = nullptr;
    try {
      request_ptr = cista::deserialize<RpcRequestHeader, MODE>(request_buffer);
      if (!request_ptr) {
        throw std::runtime_error("Failed to deserialize request");
      }

      auto it = functions_.find(request_ptr->function_id);
      if (it == functions_.end()) {
        throw std::runtime_error("Function not found");
      }

      return it->second(*request_ptr, std::move(context));
    } catch (const std::exception& e) {
      RpcResponseHeader error_response;
      if (request_ptr) {
        error_response.session_id = request_ptr->session_id;
      }
      error_response.status = std::make_error_code(std::errc::invalid_argument);
      return {std::move(error_response), std::any{}};
    }
  }

 public:
  /**
   * @brief Serializes an RpcResponseHeader into a byte buffer.
   * @param response_header The response header to serialize.
   * @return A cista::byte_buf containing the serialized data.
   */
  static cista::byte_buf serialize_response(
    RpcResponseHeader& response_header) {
    return cista::serialize<MODE>(response_header);
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
  void register_function(
    function_id_t id, Func&& func,
    const data::string& name = data::string{"anonymous"}) {
    functions_[id] = make_rpc_wrapper(std::forward<Func>(func));
    signatures_[id] = make_rpc_signature<Func>(id, name, instance_name_);
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
  template <typename ReturnContextT = std::monostate, typename InputData>
  RpcInvokeResult<ReturnContextT> dispatch(
    cista::byte_buf&& request_buffer, InputData& input_data) {
    auto [header, context_any] = dispatch_impl(
      std::move(request_buffer),
      // RpcContextPtr is a observation, it does not own the data
      RpcContextPtr(&input_data, [](void*) { /* non-owning */ }));

    ReturnContextT context = context_any.has_value()
                               ? std::any_cast<ReturnContextT>(context_any)
                               : ReturnContextT{};
    return {std::move(header), std::move(context)};
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
  template <typename ReturnContextT = std::monostate, typename InputData>
  RpcInvokeResult<ReturnContextT> dispatch_move(
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

    auto [header, context_any] =
      dispatch_impl(std::move(request_buffer), std::move(non_owning_ptr));

    ReturnContextT context = context_any.has_value()
                               ? std::any_cast<ReturnContextT>(context_any)
                               : ReturnContextT{};
    return {std::move(header), std::move(context)};
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
  template <typename ReturnContextT = std::monostate>
  RpcInvokeResult<ReturnContextT> dispatch(cista::byte_buf&& request_buffer) {
    auto [header, context_any] = dispatch_impl(
      std::move(request_buffer), RpcContextPtr(nullptr, [](void*) {}));

    ReturnContextT context = context_any.has_value()
                               ? std::any_cast<ReturnContextT>(context_any)
                               : ReturnContextT{};
    return {std::move(header), std::move(context)};
  }

  /**
   * @brief Checks if a function with the given ID is registered.
   * @param id The function ID to check.
   * @return True if the function is registered, false otherwise.
   */
  bool is_registered(function_id_t id) const {
    return functions_.find(id) != functions_.end();
  }

  /**
   * @brief Gets the total number of registered functions.
   * @return The number of functions.
   */
  size_t function_count() const { return functions_.size(); }

  /**
   * @brief Retrieves the signature of a registered function.
   * @param id The ID of the function.
   * @return An std::optional<RpcFunctionSignature> containing the signature if
   * found, otherwise std::nullopt.
   */
  std::optional<RpcFunctionSignature> get_signature(function_id_t id) const {
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
  cista::byte_buf get_all_signatures() const {
    data::vector<RpcFunctionSignature> sig_vec;
    for (const auto& pair : signatures_) {
      sig_vec.push_back(pair.second);
    }
    return cista::serialize<MODE>(sig_vec);
  }

 private:
  template <typename T>
  static constexpr ParamType get_param_type() {
    using DecayedT = std::decay_t<T>;
    constexpr bool is_arithmetic_or_enum =
      std::is_arithmetic_v<DecayedT> || std::is_enum_v<DecayedT>;

    if constexpr (std::is_same_v<DecayedT, RpcRequestHeader>) {
      return ParamType::CUSTOM;  // Should be filtered by caller
    } else if constexpr (is_arithmetic_or_enum) {
      return get_primitive_param_info<DecayedT>().type;
    } else if constexpr (is_cista_strong<DecayedT>::value) {
      using UnderlyingType =
        typename get_cista_strong_underlying_type<DecayedT>::type;
      return get_primitive_param_info<UnderlyingType>().type;
    } else if constexpr (std::is_same_v<DecayedT, data::string>) {
      return ParamType::STRING;
    } else if constexpr (is_data_vector<DecayedT>::value) {
      using ElementType = typename DecayedT::value_type;
      return get_vector_param_info<ElementType>().type;
    } else {
      return ParamType::CUSTOM;  // Context type
    }
  }

  template <typename Tuple, typename ContextType, size_t Index = 0>
  static void fill_param_types_impl(data::vector<ParamType>& params) {
    if constexpr (Index < std::tuple_size_v<Tuple>) {
      using T = std::tuple_element_t<Index, Tuple>;
      using DecayedT = std::decay_t<T>;
      if constexpr (
        !std::is_same_v<DecayedT, ContextType>
        && !std::is_same_v<DecayedT, RpcRequestHeader>) {
        params.push_back(get_param_type<T>());
      }
      fill_param_types_impl<Tuple, ContextType, Index + 1>(params);
    }
  }

  template <typename Func>
  static RpcFunctionSignature make_rpc_signature(
    function_id_t id, const data::string& function_name,
    const data::string& instance_name) {
    using Traits = function_traits<std::decay_t<Func>>;
    using ReturnType = typename Traits::return_type;
    using ArgsTuple = typename Traits::args_tuple;
    using ContextType = typename context_finder<ArgsTuple>::type;

    RpcFunctionSignature sig;
    sig.instance_name = instance_name;
    sig.id = id;
    sig.function_name = function_name;
    sig.takes_context = !std::is_void_v<ContextType>;

    fill_param_types_impl<ArgsTuple, ContextType>(sig.param_types);

    using DecayR = std::decay_t<ReturnType>;
    if constexpr (std::is_void_v<DecayR>) {
      sig.return_type = ParamType::VOID;
    } else {
      static constexpr bool is_rpc_header_pair =
        is_pair<DecayR>::value
        && is_rpc_response_header<
          std::decay_t<typename get_pair_first_type<DecayR>::type>>::value;
      static constexpr bool is_arithmetic_or_enum =
        std::is_arithmetic_v<DecayR> || std::is_enum_v<DecayR>;
      static constexpr bool is_serializable_value =
        is_arithmetic_or_enum || std::is_same_v<DecayR, data::string>
        || is_data_vector<DecayR>::value;

      if constexpr (std::is_same_v<DecayR, RpcResponseHeader>) {
        sig.return_type = ParamType::CUSTOM;
      } else if constexpr (is_rpc_header_pair) {
        sig.return_type = ParamType::CUSTOM;
      } else if constexpr (is_serializable_value) {
        sig.return_type = get_param_type<DecayR>();
      } else {
        sig.return_type = ParamType::CUSTOM;
      }
    }
    return sig;
  }

  template <typename Func>
  static ErasedRpcFunction make_rpc_wrapper(Func&& func) {
    return [func = std::forward<Func>(func)](
             RpcRequestHeader& request, RpcContextPtr context_ptr)
             -> std::pair<RpcResponseHeader, std::any> {
      try {
        using Traits = function_traits<std::decay_t<Func>>;
        using ReturnType = typename Traits::return_type;
        using ArgsTuple = typename Traits::args_tuple;
        using ContextType = typename context_finder<ArgsTuple>::type;

        auto invoke_and_package =
          [&]() -> std::pair<RpcResponseHeader, std::any> {
          auto args = [&]() {
            if constexpr (std::is_void_v<ContextType>) {
              return extract_args_impl<ArgsTuple>(
                request, static_cast<void*>(nullptr),
                std::make_index_sequence<Traits::arity>{});
            } else {
              auto* typed_context =
                static_cast<ContextType*>(context_ptr.get());
              if (!typed_context) {
                throw std::runtime_error("Context required but not provided.");
              }
              return extract_args_impl<ArgsTuple>(
                request, typed_context,
                std::make_index_sequence<Traits::arity>{});
            }
          }();

          RpcResponseHeader response;
          response.session_id = request.session_id;

          if constexpr (std::is_void_v<ReturnType>) {
            // Case: Function returns void
            std::apply(func, std::move(args));
            return {std::move(response), std::any{}};
          } else {
            // Case: Function returns a value
            auto result = std::apply(func, std::move(args));
            using DecayR = std::decay_t<decltype(result)>;

            // Use a constexpr lambda to process the result based on its type.
            // This can sometimes help the compiler with complex template logic.
            return [&]() -> std::pair<RpcResponseHeader, std::any> {
              static constexpr bool is_rpc_header_pair =
                is_pair<DecayR>::value
                && is_rpc_response_header<std::decay_t<
                  typename get_pair_first_type<DecayR>::type>>::value;
              static constexpr bool is_arithmetic_or_enum =
                std::is_arithmetic_v<DecayR> || std::is_enum_v<DecayR>;
              static constexpr bool is_serializable_value =
                is_arithmetic_or_enum || std::is_same_v<DecayR, data::string>
                || is_data_vector<DecayR>::value;

              if constexpr (std::is_same_v<DecayR, RpcResponseHeader>) {
                // Returns RpcResponseHeader directly
                return {std::move(result), std::any{}};
              } else if constexpr (is_rpc_header_pair) {
                // Returns std::pair<RpcResponseHeader, Any>
                return {
                  std::move(result.first), std::any(std::move(result.second))};
              } else if constexpr (is_serializable_value) {
                // Returns a serializable value
                ParamMeta result_meta;
                if constexpr (is_arithmetic_or_enum) {
                  result_meta.type = get_primitive_param_info<DecayR>().type;
                  result_meta.value.emplace<PrimitiveValue>(std::move(result));
                } else if constexpr (std::is_same_v<DecayR, data::string>) {
                  result_meta.type = ParamType::STRING;
                  result_meta.value.emplace<data::string>(std::move(result));
                } else {  // is_data_vector
                  result_meta.type =
                    get_vector_param_info<typename DecayR::value_type>().type;
                  result_meta.value.emplace<VectorValue>(std::move(result));
                }
                response.results.emplace_back(std::move(result_meta));
                return {std::move(response), std::any{}};
              } else {
                // Returns a context object
                return {std::move(response), std::any(std::move(result))};
              }
            }();
          }
        };

        return invoke_and_package();
      } catch (const std::exception& e) {
        RpcResponseHeader error_response;
        error_response.session_id = request.session_id;
        error_response.status =
          std::make_error_code(std::errc::invalid_argument);
        // In a real application, you would log e.what() here.
        return {std::move(error_response), std::any{}};
      }
    };
  }
};

}  // namespace rpc_core
}  // namespace stdexe_ucx_runtime

#endif  // RPC_CORE_RPC_DISPATCHER_HPP_
