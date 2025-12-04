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

#ifndef RPC_CORE_RPC_ASYNC_DISPATCHER_HPP_
#define RPC_CORE_RPC_ASYNC_DISPATCHER_HPP_

#include <cista.h>
#include <proxy/proxy.h>

#include <exception>
#include <utility>

#include <unifex/any_sender_of.hpp>
#include <unifex/just.hpp>
#include <unifex/just_error.hpp>
#include <unifex/just_from.hpp>
#include <unifex/let_error.hpp>
#include <unifex/let_value.hpp>
#include <unifex/let_value_with.hpp>
#include <unifex/sender_concepts.hpp>
#include <unifex/then.hpp>

#include "rpc_core/rpc_dispatcher.hpp"

namespace eux {
namespace rpc {

/**
 * @brief The type of the result of an asynchronous RPC invocation.
 *
 * @tparam ContextT The type of the context object. Defaults to std::monostate
 * if no context is returned.
 */
template <typename ContextT = std::monostate>
using RpcInvokeAsyncResult = unifex::any_sender_of<RpcInvokeResult<ContextT>>;

struct ErasedAsyncRpcFunctionFacade
  : pro::facade_builder  //
    ::add_convention<
      pro::operator_dispatch<"()">,
      unifex::any_sender_of<std::pair<RpcResponseHeader, ReturnedPayload>>(
        const RpcRequestHeader&, RpcContextPtr, const RpcResponseBuilder&)>  //
    ::support_relocation<pro::constraint_level::trivial>                     //
    ::build {};

using ErasedAsyncRpcFunction = pro::proxy<ErasedAsyncRpcFunctionFacade>;

namespace detail {
// Define function signatures in pro::proxy Facade for dynamic RPC
using DynamicAsyncFunctionReturnType =
  unifex::any_sender_of<std::pair<data::vector<ParamMeta>, ReturnedPayload>>;
template <typename PayloadT>
using DynamicAsyncConventionSignature = DynamicAsyncFunctionReturnType(
  const data::vector<ParamMeta>&, const PayloadT&);
using DynamicAsyncConventionSignatureNoPayload =
  DynamicAsyncFunctionReturnType(const data::vector<ParamMeta>&);

// Define initial builder (contains "no payload" base version)
using InitialAsyncBuilder = pro::facade_builder  //
  ::add_skill<pro::skills::as_view>              //
  ::add_convention<
    pro::weak_dispatch<RpcCallOperator>,
    DynamicAsyncConventionSignatureNoPayload>;
// Use TMP helper template to add all Payload conventions
using BuilderWithAsyncPayloads = AddVariantConventionsType<
  InitialAsyncBuilder, PayloadVariant, DynamicAsyncConventionSignature,
  RpcCallOperator>;
}  // namespace detail

// Final build Facade
struct DynamicAsyncRpcFunctionFacade
  : detail::BuilderWithAsyncPayloads                               //
    ::template support_relocation<pro::constraint_level::trivial>  //
    ::build {};

using DynamicAsyncRpcFunction = pro::proxy<DynamicAsyncRpcFunctionFacade>;
using DynamicAsyncRpcFunctionView =
  pro::proxy_view<DynamicAsyncRpcFunctionFacade>;

// Helper template to check if a sender returns void
template <typename SenderValueTypes>
struct is_void_sender_helper {
  static constexpr bool value = false;
};

template <
  template <typename...> class Variant, template <typename...> class Tuple>
struct is_void_sender_helper<unifex::type_list<Variant<Tuple<>>>> {
  static constexpr bool value = true;
};

// Helper to check if a sender type returns void without instantiating
// problematic templates
template <typename Sender>
struct is_void_sender_type {
  static constexpr bool value = false;
};

// Specialization for any_sender_of<void>
template <>
struct is_void_sender_type<unifex::any_sender_of<void>> {
  static constexpr bool value = true;
};

template <typename Sender>
struct is_empty_sender_type {
  static constexpr bool value = false;
};

// Specialization for any_sender_of<>
template <>
struct is_empty_sender_type<unifex::any_sender_of<>> {
  static constexpr bool value = true;
};

// SFINAE helper to extract value types safely
template <typename Sender, typename = void>
struct safe_sender_value_types {
  using type = void;
  static constexpr bool is_void = true;
};

template <typename Sender>
struct safe_sender_value_types<
  Sender, std::void_t<unifex::sender_value_types_t<
            Sender, unifex::type_list, unifex::type_list>>> {
  using type =
    unifex::sender_value_types_t<Sender, unifex::type_list, unifex::type_list>;
  static constexpr bool is_void = is_void_sender_helper<type>::value;
};

class AsyncRpcDispatcher : public RpcDispatcher {
 public:
  /**
   * @brief Constructs an AsyncRpcDispatcher with a unique instance name.
   *
   * @param instance_name A name to identify this dispatcher instance, used in
   * function signatures for service discovery.
   */
  explicit AsyncRpcDispatcher(data::string instance_name)
    : RpcDispatcher(std::move(instance_name)) {}

  /**
   * @brief Invokes an asynchronous RPC function and returns a sender.
   *
   * @param request The request header.
   * @param context The context pointer.
   * @param builder The response builder.
   * @return A sender that completes with the response header and payload.
   */
  unifex::any_sender_of<std::pair<RpcResponseHeader, ReturnedPayload>>
  InvokeAsyncFunction(
    const RpcRequestHeader& request, RpcContextPtr context,
    const RpcResponseBuilder& builder) {
    auto it = async_functions_.find(request.function_id);
    if (it == async_functions_.end()) {
      return unifex::just_error(
        MakeRpcExceptionPtr(RpcErrc::NOT_FOUND, "Function not found"));
    }
    try {
      // pro::proxy may throw error
      return (*it->second)(request, std::move(context), builder);
    } catch (const std::exception& e) {
      return unifex::just_error(
        MakeRpcExceptionPtr(RpcErrc::INVALID_ARGUMENT, e.what()));
    }
  }

  /**
   * @brief Registers a callable that returns a sender as an RPC function.
   *
   * @tparam Func The type of the callable that returns a sender.
   * @param id The unique identifier for this function.
   * @param func The callable to register.
   * @param name A human-readable name for the function, used for service
   * discovery. Defaults to "anonymous".
   */
  template <typename Func>
  void RegisterFunction(
    function_id_t id, Func&& func,
    const data::string& name = data::string{"anonymous"}) {
    async_functions_.emplace(id, MakeAsyncRpcWrapper(std::forward<Func>(func)));

    signatures_.emplace(id, MakeRpcSignature<Func>(id, name, instance_name_));
  }

  /**
   * @brief Registers a type-erased dynamic async function as an RPC function.
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
   * an `InputPayloadVariant` and returns a sender of a pair of a vector of
   * result `ParamMeta` and a `ReturnedPayload`.
   */
  void RegisterFunction(
    function_id_t id, const data::string& name,
    const data::vector<ParamType>& param_types,
    const data::vector<ParamType>& return_types, PayloadType input_payload_type,
    PayloadType return_payload_type, DynamicAsyncRpcFunction&& func) {
    async_functions_.emplace(
      id, pro::make_proxy<ErasedAsyncRpcFunctionFacade>(
            DynamicAsyncFuncWrapper<DynamicAsyncRpcFunction>{
              std::move(func), input_payload_type}));

    // Create and validate the signature manually from the provided types.
    RpcFunctionSignature sig{
      .instance_name = instance_name_,
      .id = id,
      .function_name = name,
      .param_types = param_types,
      .return_types = return_types,
      .input_payload_type = input_payload_type,
      .return_payload_type = return_payload_type,
    };

    // Validate tensor and payload constraints
    RpcDispatcher::ValidateTensorPayloadConstraints(sig);

    signatures_.emplace(id, std::move(sig));
  }

  /**
   * @brief Deserializes a request, invokes the corresponding async function
   * with a context passed by reference, and returns a sender.
   *
   * @tparam ReturnContextT The expected type of the context returned by the
   * function.
   * @tparam InputData The type of the context data to be passed to the
   * function.
   * @param request_buffer The serialized request buffer.
   * @param input_data A reference to the context data.
   * @return A sender that completes with RpcInvokeResult containing the
   * response header and any returned context.
   */
  template <typename ReturnContextT, typename InputData>
  RpcInvokeAsyncResult<ReturnContextT> Dispatch(
    cista::byte_buf&& request_buffer, InputData& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_buffer = std::move(request_buffer)]() mutable {
        return request_buffer;
      },
      [this, &input_data, &builder](cista::byte_buf& request_buffer) {
        RpcContextPtr non_owning_ptr(
          const_cast<InputData*>(&input_data), [](void*) { /* non-owning */ });
        return DispatchImpl<ReturnContextT>(
          request_buffer, std::move(non_owning_ptr), builder);
      });
  }

  template <typename ReturnContextT, typename InputData>
  RpcInvokeAsyncResult<ReturnContextT> Dispatch(
    const RpcRequestHeader& request_header, InputData& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return DispatchImpl<ReturnContextT>(
      request_header, RpcContextPtr(&input_data, [](void*) {}), builder);
  }

  /**
   * @brief Deserializes a request, invokes the corresponding async function by
   * moving a context into it, and returns a sender.
   *
   * @tparam ReturnContextT The expected type of the context returned by the
   * function.
   * @tparam InputData The type of the context data to be moved.
   * @param request_buffer The serialized request buffer.
   * @param input_data An rvalue reference to the context data.
   * @return A sender that completes with RpcInvokeResult containing the
   * response header and any returned context.
   */
  template <typename ReturnContextT, typename InputData>
  RpcInvokeAsyncResult<ReturnContextT> DispatchMove(
    cista::byte_buf&& request_buffer, InputData&& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_buffer = std::move(request_buffer),
       input_data = std::move(input_data)]() mutable {
        return std::make_tuple(
          std::move(request_buffer), std::move(input_data));
      },
      [this, &builder](auto& pair) {
        auto& [request_buffer, input_data] = pair;
        RpcContextPtr non_owning_ptr(&input_data, [](void*) {});
        return DispatchImpl<ReturnContextT>(
          request_buffer, std::move(non_owning_ptr), builder);
      });
  }

  template <typename ReturnContextT, typename InputData>
  RpcInvokeAsyncResult<ReturnContextT> DispatchMove(
    RpcRequestHeader&& request_header, InputData&& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_header = std::move(request_header),
       input_data = std::move(input_data)]() mutable {
        return std::make_tuple(
          std::move(request_header), std::move(input_data));
      },
      [this, &builder](auto& pair) {
        auto& [request_header, input_data] = pair;
        RpcContextPtr non_owning_ptr(&input_data, [](void*) {});
        return DispatchImpl<ReturnContextT>(
          request_header, std::move(non_owning_ptr), builder);
      });
  }

  /**
   * @brief Deserializes a request, invokes an async function that takes no
   * context, and returns a sender.
   *
   * @tparam ReturnContextT The expected type of the context returned by the
   * function.
   * @param request_buffer The serialized request buffer.
   * @return A sender that completes with RpcInvokeResult containing the
   * response header and any returned context.
   */
  template <typename ReturnContextT>
  RpcInvokeAsyncResult<ReturnContextT> Dispatch(
    cista::byte_buf&& request_buffer,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_buffer = std::move(request_buffer)]() mutable {
        return request_buffer;
      },
      [this, &builder](cista::byte_buf& request_buffer) {
        return DispatchImpl<ReturnContextT>(
          request_buffer, RpcContextPtr(nullptr, [](void*) {}), builder);
      });
  }

  template <typename ReturnContextT>
  RpcInvokeAsyncResult<ReturnContextT> Dispatch(
    const RpcRequestHeader& request_header,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return DispatchImpl<ReturnContextT>(
      request_header, RpcContextPtr(nullptr, [](void*) {}), builder, nullptr);
  }

  /**
   * @brief Dynamically dispatches a request without knowing the return context
   * type at compile time, returning a sender.
   *
   * @param request_buffer The serialized request buffer.
   * @return A sender that completes with RpcInvokeResult containing the header
   * and an optional variant of the returned context.
   */
  RpcInvokeAsyncResult<ReturnedPayload> Dispatch(
    cista::byte_buf&& request_buffer,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_buffer = std::move(request_buffer)]() mutable {
        return request_buffer;
      },
      [this, &builder](cista::byte_buf& request_buffer) {
        return DispatchImpl(
          request_buffer, RpcContextPtr(nullptr, [](void*) {}), builder);
      });
  }

  RpcInvokeAsyncResult<ReturnedPayload> Dispatch(
    RpcRequestHeader&& request_header,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_header = std::move(request_header)]() mutable {
        return request_header;
      },
      [this, &builder](RpcRequestHeader& request_header) {
        return DispatchImpl(
          request_header, RpcContextPtr(nullptr, [](void*) {}), builder);
      });
  }

  template <typename InputData>
  RpcInvokeAsyncResult<ReturnedPayload> Dispatch(
    cista::byte_buf&& request_buffer, const InputData& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_buffer = std::move(request_buffer)]() mutable {
        return request_buffer;
      },
      [this, &input_data, &builder](cista::byte_buf& request_buffer) {
        RpcContextPtr non_owning_ptr(
          const_cast<InputData*>(&input_data), [](void*) { /* non-owning */ });
        return DispatchImpl(request_buffer, std::move(non_owning_ptr), builder);
      });
  }

  template <typename InputData>
  RpcInvokeAsyncResult<ReturnedPayload> Dispatch(
    const RpcRequestHeader& request_header, const InputData& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return DispatchImpl(
      request_header,
      RpcContextPtr(const_cast<InputData*>(&input_data), [](void*) {}),
      builder);
  }

  template <typename InputData>
  RpcInvokeAsyncResult<ReturnedPayload> DispatchMove(
    cista::byte_buf&& request_buffer, InputData&& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_buffer = std::move(request_buffer),
       input_data = std::move(input_data)]() mutable {
        return std::make_tuple(
          std::move(request_buffer), std::move(input_data));
      },
      [this, &builder](auto& pair) {
        auto& [request_buffer, input_data] = pair;
        RpcContextPtr non_owning_ptr(&input_data, [](void*) {});
        return DispatchImpl(request_buffer, std::move(non_owning_ptr), builder);
      });
  }

  template <typename InputData>
  RpcInvokeAsyncResult<ReturnedPayload> DispatchMove(
    RpcRequestHeader&& request_header, InputData&& input_data,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [request_header = std::move(request_header),
       input_data = std::move(input_data)]() mutable {
        return std::make_tuple(
          std::move(request_header), std::move(input_data));
      },
      [this, &builder](auto& pair) {
        auto& [request_header, input_data] = pair;
        RpcContextPtr non_owning_ptr(&input_data, [](void*) {});
        return DispatchImpl(request_header, std::move(non_owning_ptr), builder);
      });
  }

  template <typename Func>
  unifex::any_sender_of<std::pair<RpcResponseHeader, ReturnedPayload>>
  DispatchAdhoc(
    Func&& func, RpcRequestHeader&& header, RpcContextPtr context,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [this, header = std::move(header),
       func = std::forward<Func>(func)]() mutable {
        auto wrapper = MakeAsyncRpcWrapper(std::forward<Func>(func));
        return std::make_tuple(std::move(header), std::move(wrapper));
      },
      [this, &builder, context = std::move(context)](auto& pair) mutable {
        auto& [header, wrapper] = pair;
        return (*wrapper)(header, std::move(context), builder);
      });
  }

  unifex::any_sender_of<std::pair<RpcResponseHeader, ReturnedPayload>>
  DispatchAdhoc(
    DynamicAsyncRpcFunctionView func, PayloadType input_payload_type,
    const RpcRequestHeader& header, RpcContextPtr context,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return unifex::let_value_with(
      [func, input_payload_type]() mutable {
        auto wrapper = pro::make_proxy<ErasedAsyncRpcFunctionFacade>(
          DynamicAsyncFuncWrapper<DynamicAsyncRpcFunctionView>{
            func, input_payload_type});
        return wrapper;
      },
      [&builder, &header, context = std::move(context)](
        pro::proxy<ErasedAsyncRpcFunctionFacade>& wrapper) mutable {
        return (*wrapper)(header, std::move(context), builder);
      });
  }

  /**
   * @brief Checks if a function with the given ID is registered.
   * @param id The function ID to check.
   * @return True if the function is registered, false otherwise.
   */
  bool IsRegistered(function_id_t id) const {
    return async_functions_.find(id) != async_functions_.end();
  }

  /**
   * @brief Gets the total number of registered functions.
   * @return The number of functions.
   */
  size_t FunctionCount() const { return async_functions_.size(); }

 protected:
  cista::raw::hash_map<function_id_t, ErasedAsyncRpcFunction> async_functions_;

  template <typename Func, typename PayloadT>
  static constexpr bool is_dynamic_async_function_v = requires(Func&& func) {
    {
      (*std::forward<Func>(func))(
        std::declval<const data::vector<ParamMeta>&>(),
        std::declval<const PayloadT&>())
    };  // NOLINT(readability/braces)
  };

  template <typename Func>
  struct DynamicAsyncFuncWrapper {
    Func dynamic_func;
    PayloadType input_payload_type;

    unifex::any_sender_of<std::pair<RpcResponseHeader, ReturnedPayload>>
    operator()(
      const RpcRequestHeader& request, RpcContextPtr context_ptr,
      const RpcResponseBuilder& builder) {
      constexpr bool is_reference_wrapper =
        is_reference_wrapper_v<std::decay_t<Func>>;
      constexpr bool is_dynamic_function =
        is_dynamic_async_function_v<Func, ucxx::UcxBuffer>
        || is_dynamic_async_function_v<Func, ucxx::UcxBufferVec>;
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
      try {
        return call_fn() | unifex::then([&request](auto&& result) {
                 auto& [serializable_results, returned_payload] = result;
                 RpcResponseHeader response_header;
                 response_header.session_id = request.session_id;
                 response_header.request_id = request.request_id;
                 response_header.status =
                   std::make_error_code(rpc::RpcErrc::OK);
                 response_header.results = std::move(serializable_results);
                 return std::make_pair(
                   std::move(response_header), std::move(returned_payload));
               });
      } catch (const std::exception& e) {
        return unifex::just_error(
          MakeRpcExceptionPtr(RpcErrc::INVALID_ARGUMENT, e.what()));
      }
    }
  };

  template <typename ReturnContextT>
    requires(
      !std::is_same_v<ReturnContextT, void>
      && !std::is_same_v<ReturnContextT, ReturnedPayload>)
  RpcInvokeAsyncResult<ReturnContextT> DispatchImpl(
    const RpcRequestHeader& request_header, RpcContextPtr context,
    const RpcResponseBuilder& builder = RpcResponseBuilder{},
    ReturnContextT* = nullptr) {
    return InvokeAsyncFunction(request_header, std::move(context), builder)
           | unifex::then([this](auto&& header_context_pair) {
               auto& [header, context_variant] = header_context_pair;
               return BuildRpcResult<ReturnContextT>(
                 std::move(header), std::move(context_variant));
             });
  }

  template <typename ReturnContextT>
    requires(
      !std::is_same_v<ReturnContextT, void>
      && !std::is_same_v<ReturnContextT, ReturnedPayload>)
  RpcInvokeAsyncResult<ReturnContextT> DispatchImpl(
    const cista::byte_buf& request_buffer, RpcContextPtr context,
    const RpcResponseBuilder& builder = RpcResponseBuilder{},
    ReturnContextT* = nullptr) {
    const RpcRequestHeader* request_ptr =
      cista::deserialize<const RpcRequestHeader, utils::SerializerMode>(
        request_buffer);
    if (!request_ptr) {
      return unifex::just_error(MakeRpcExceptionPtr(
        RpcErrc::UNAVAILABLE, "Failed to deserialize request"));
    }
    return DispatchImpl<ReturnContextT>(
      *request_ptr, std::move(context), builder, nullptr);
  }

  RpcInvokeAsyncResult<ReturnedPayload> DispatchImpl(
    const RpcRequestHeader& request_header, RpcContextPtr context,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    return InvokeAsyncFunction(request_header, std::move(context), builder)
           | unifex::then([](auto&& header_context_pair) mutable {
               auto& [header, context_variant] = header_context_pair;
               return RpcInvokeResult<ReturnedPayload>{
                 std::move(header), std::move(context_variant)};
             });
  }

  RpcInvokeAsyncResult<ReturnedPayload> DispatchImpl(
    const cista::byte_buf& request_buffer, RpcContextPtr context,
    const RpcResponseBuilder& builder = RpcResponseBuilder{}) {
    const RpcRequestHeader* request_ptr =
      cista::deserialize<const RpcRequestHeader, utils::SerializerMode>(
        request_buffer);
    if (!request_ptr) {
      return unifex::just_error(MakeRpcExceptionPtr(
        RpcErrc::UNAVAILABLE, "Failed to deserialize request"));
    }
    return DispatchImpl(*request_ptr, std::move(context), builder);
  }

  template <typename Func>
  pro::proxy<ErasedAsyncRpcFunctionFacade> MakeAsyncRpcWrapper(Func&& func) {
    return pro::make_proxy<ErasedAsyncRpcFunctionFacade>(
      [this, func = std::forward<Func>(func)](
        const RpcRequestHeader& request, RpcContextPtr context_ptr,
        const RpcResponseBuilder& builder)
        -> unifex::any_sender_of<
          std::pair<RpcResponseHeader, ReturnedPayload>> {
        using Traits = function_traits<std::decay_t<Func>>;
        using ReturnSenderType = typename Traits::return_type;
        using ArgsTuple = typename Traits::args_tuple;
        using ContextType = typename payload_finder<ArgsTuple>::type;

        // Check if function returns a sender
        static_assert(
          unifex::sender<ReturnSenderType>,
          "Async function must return a sender type");

        // Extract arguments
        // Use payload_finder to find context type (same as synchronous
        // version) All context types should be passed as references, not
        // pointers
        auto args_fn = [](
                         const RpcRequestHeader& request_,
                         const RpcContextPtr& context_ptr_,
                         const RpcResponseBuilder& builder_) {
          if constexpr (std::is_void_v<ContextType>) {
            return extract_args_impl<ArgsTuple>(
              request_, static_cast<void*>(nullptr), builder_,
              std::make_index_sequence<Traits::arity>{});
          } else {
            auto* typed_context = static_cast<ContextType*>(context_ptr_.get());
            if (!typed_context) {
              throw std::runtime_error("Context required but not provided.");
            }
            return extract_args_impl<ArgsTuple>(
              request_, typed_context, builder_,
              std::make_index_sequence<Traits::arity>{});
          }
        };

        // Invoke function and get sender
        // auto result_sender =
        //   unifex::just_from([&, context_ptr_ = std::move(context_ptr)]() {
        //     auto args = args_fn(request, context_ptr_, builder);
        //     return std::apply(func, std::move(args));
        //   });
        auto args = args_fn(request, context_ptr, builder);
        auto result_sender = std::apply(func, std::move(args));

        // Check if sender completes with void
        // First check if ReturnSenderType is any_sender_of<void>
        constexpr bool is_void_sender =
          is_void_sender_type<ReturnSenderType>::value
          || safe_sender_value_types<ReturnSenderType>::is_void;
        static_assert(
          !is_void_sender, "Function must not return a sender of void");

        constexpr bool is_empty_sender =
          is_empty_sender_type<ReturnSenderType>::value;

        // Transform the sender result to match expected signature
        if constexpr (is_empty_sender) {
          return std::move(result_sender)
                 | unifex::then([&request, &builder]() {
                     auto header = builder.PrepareResponse(
                       request.session_id, request.request_id);
                     return std::make_pair(
                       std::move(header), ReturnedPayload{std::monostate{}});
                   });
        } else {
          // Handle non-void return types
          auto resp_sender = unifex::then([&request, &builder](auto&& result) {
            using DecayR = std::decay_t<decltype(result)>;

            // Handle different return types
            if constexpr (std::is_same_v<DecayR, RpcResponseHeader>) {
              return std::make_pair(
                std::move(result), ReturnedPayload{std::monostate{}});
            } else {
              // Check if it's a pair with RpcResponseHeader
              static constexpr bool is_header_pair =
                is_pair<DecayR>::value
                && std::is_same_v<
                  std::decay_t<typename get_pair_first_type<DecayR>::type>,
                  RpcResponseHeader>;

              if constexpr (is_header_pair) {
                return std::make_pair(
                  std::move(result.first),
                  ReturnedPayload(std::move(result.second)));
              } else {
                // Automatic mode: Build response from the function's
                // return value
                // Capture result by value to ensure it's properly copied
                auto builder_result = [&request, &builder,
                                       result_value = std::move(result)]() {
                  if constexpr (is_std_tuple<DecayR>::value) {
                    return std::apply(
                      [&request, &builder](auto&&... results) {
                        return builder.PrepareResponse(
                          request.session_id, request.request_id,
                          std::forward<decltype(results)>(results)...);
                      },
                      std::move(result_value));
                  } else {
                    return builder.PrepareResponse(
                      request.session_id, request.request_id,
                      std::move(result_value));
                  }
                }();

                if constexpr (std::is_same_v<
                                std::decay_t<decltype(builder_result)>,
                                RpcResponseHeader>) {
                  return std::make_pair(
                    std::move(builder_result),
                    ReturnedPayload{std::monostate{}});
                } else {
                  return std::make_pair(
                    std::move(builder_result.first),
                    std::move(builder_result.second));
                }
              }
            }
          });
          return std::move(result_sender) | std::move(resp_sender);
        }
      });
  }
};

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_ASYNC_DISPATCHER_HPP_
