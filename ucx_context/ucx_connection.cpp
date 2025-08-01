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

#include "ucx_context/ucx_connection.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <ucs/sys/sock.h>

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace stdexe_ucx_runtime {

/*
TODO(He Jia): For now only support Active Message.
Please add Tag Message support in the future.
*/

// Static member initialization
std::uint32_t UcxConnection::num_instances_ = 0;

// EmptyCallback implementation
void EmptyCallback::operator()(ucs_status_t status) {
  // Do nothing
}

EmptyCallback* EmptyCallback::get() {
  static EmptyCallback instance;
  return &instance;
}

std::unique_ptr<EmptyCallback> EmptyCallback::get_unique() {
  return std::unique_ptr<EmptyCallback>(new EmptyCallback());
}

// UcxCallback destructor
UcxCallback::~UcxCallback() {}

// CqeEntryCallback implementation
void CqeEntryCallback::operator()(ucs_status_t status) {
  auto& entry = get_entry_fn_();
  entry.user_data = user_data_;
  entry.res = status;
}

// UcxConnection implementation
UcxConnection::UcxConnection(
  ucp_worker_h worker,
  std::unique_ptr<UcxCallback>
    handle_err_cb,
  std::function<std::uint64_t()>
    get_conn_id)
  : worker_(worker),
    handle_err_cb_(std::move(handle_err_cb)),
    establish_cb_(nullptr),
    disconnect_cb_(nullptr),
    conn_id_(get_conn_id()),
    remote_conn_id_(0),
    ep_(nullptr),
    close_request_(nullptr),
    ucx_status_(UCS_INPROGRESS) {
  ++num_instances_;
  struct sockaddr_in in_addr = {0};
  in_addr.sin_family = AF_INET;
  set_log_prefix((const struct sockaddr*)&in_addr, sizeof(in_addr));
  ucs_list_head_init(&all_requests_);
  UCX_CONN_DEBUG << "created new connection " << this
                 << " total: " << num_instances_ << "\n";
}

UcxConnection::~UcxConnection() {
  /* establish cb must be destroyed earlier since it accesses
   * the connection */
  if (ep_ != nullptr) {
    UCX_CONN_ERROR << "Disconnect connection before destroying! closing ep "
                   << ep_ << " mode force"
                   << "\n";
    ucp_ep_close_nb(ep_, UCP_EP_CLOSE_MODE_FORCE);
  }
  assert(ep_ == nullptr);
  assert(ucs_list_is_empty(&all_requests_));
  assert(!UCS_PTR_IS_PTR(close_request_));

  UCX_CONN_DEBUG << "UcxConnection destroyed"
                 << "\n";
  --num_instances_;
}

void UcxConnection::connect(
  const struct sockaddr* src_saddr, const struct sockaddr* dst_saddr,
  socklen_t addrlen, std::unique_ptr<UcxCallback> callback) {
  set_log_prefix(dst_saddr, addrlen);

  ucp_ep_params_t ep_params;
  ep_params.field_mask =
    UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
  ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
  ep_params.sockaddr.addr = dst_saddr;
  ep_params.sockaddr.addrlen = addrlen;
  if (src_saddr != nullptr) {
    ep_params.field_mask |= UCP_EP_PARAM_FIELD_LOCAL_SOCK_ADDR;
    ep_params.local_sockaddr.addr = src_saddr;
    ep_params.local_sockaddr.addrlen = addrlen;
  }
  if (callback->get_client_id() != CLIENT_ID_UNDEFINED) {
    ep_params.flags |= UCP_EP_PARAMS_FLAGS_SEND_CLIENT_ID;
  }

  char sockaddr_str[UCS_SOCKADDR_STRING_LEN];
  UCX_CONN_LOG << "Connecting to "
               << ucs_sockaddr_str(
                    dst_saddr, sockaddr_str, UCS_SOCKADDR_STRING_LEN)
               << "\n";

  connect_common(ep_params, std::move(callback));
}

void UcxConnection::accept(
  ucp_conn_request_h conn_req, std::unique_ptr<UcxCallback> callback) {
  ucp_conn_request_attr_t conn_req_attr;
  conn_req_attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;

  ucs_status_t status = ucp_conn_request_query(conn_req, &conn_req_attr);
  if (status == UCS_OK) {
    set_log_prefix(
      (const struct sockaddr*)&conn_req_attr.client_address,
      sizeof(conn_req_attr.client_address));
  } else {
    UCX_CONN_ERROR << "ucp_conn_request_query() failed: "
                   << ucs_status_string(status) << "\n";
  }

  ucp_ep_params_t ep_params;
  ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
  ep_params.conn_request = conn_req;
  connect_common(ep_params, std::move(callback));
}

void UcxConnection::disconnect(std::unique_ptr<UcxCallback> callback) {
  if (ep_ == nullptr) {
    return;
  }

  UCX_CONN_DEBUG << "disconnect, ep is " << ep_ << "\n";

  assert(disconnect_cb_ == nullptr);
  disconnect_cb_ = std::move(callback);

  disconnect_cb_->mark_inactive(id());

  if (!is_established()) {
    assert(ucx_status_ == UCS_INPROGRESS);
    established(UCS_ERR_CANCELED);
  } else if (ucx_status_ == UCS_OK) {
    ucx_status_ = UCS_ERR_NOT_CONNECTED;
  }

  assert(UCS_STATUS_IS_ERR(ucx_status_));

  disconnect_cb_->mark_disconnecting_from_inactive(id());

  cancel_all();

  // close the EP after cancelling all outstanding operations to purge all
  // requests scheduled on the EP which could wait for the acknowledgments
  ep_close(UCP_EP_CLOSE_MODE_FORCE);
  ep_ = nullptr;
}

void UcxConnection::disconnect_direct() {
  cancel_all();
  if (ep_) {
    ep_close(UCP_EP_CLOSE_MODE_FORCE);
  }
  ep_ = nullptr;
}

bool UcxConnection::disconnect_progress(std::unique_ptr<UcxCallback> callback) {
  disconnect_cb_ = std::move(callback);

  if (!ucs_list_is_empty(&all_requests_)) {
    return false;
  }

  if (UCS_PTR_IS_PTR(close_request_)) {
    if (ucp_request_check_status(close_request_) == UCS_INPROGRESS) {
      return false;
    } else {
      ucp_request_free(close_request_);
      close_request_ = nullptr;
    }
  }

  UCX_CONN_LOG << "disconnection completed"
               << "\n";

  invoke_callback(disconnect_cb_, UCS_OK);
  return true;
}

std::tuple<ucs_status_t, UcxRequest*> UcxConnection::send_am_data(
  const void* meta, size_t meta_length, const void* buffer, size_t length,
  ucp_mem_h memh, std::unique_ptr<UcxCallback> callback) {
  if (ep_ == nullptr) {
    (*callback)(UCS_ERR_CANCELED);
    return std::make_tuple(UCS_ERR_CANCELED, nullptr);
  }

  ucp_request_param_t param;
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_FLAGS;
  param.cb.send = (ucp_send_nbx_callback_t)common_request_callback;
  param.flags = UCP_AM_SEND_FLAG_REPLY;
  param.datatype = UCP_DATATYPE_CONTIG;
  if (memh) {
    param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
    param.memh = memh;
  }

  ucs_status_ptr_t sptr = ucp_am_send_nbx(
    ep_, DEFAULT_AM_MSG_ID, meta, meta_length, buffer, length, &param);
  return process_request(
    "ucp_am_send_nbx", sptr, std::move(callback), UcxRequestType::Send);
}

std::tuple<ucs_status_t, UcxRequest*> UcxConnection::recv_am_data(
  void* buffer, size_t length, ucp_mem_h memh, const UcxAmDesc&& data_desc,
  std::unique_ptr<UcxCallback> callback) {
  assert(ep_ != nullptr);

  if (__builtin_expect(!ucx_am_is_rndv(data_desc), false)) {
    (*callback)(UCS_OK);
    return std::make_tuple(UCS_OK, nullptr);
  }

  ucp_request_param_t param;
  param.op_attr_mask =
    UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
  param.cb.recv_am = (ucp_am_recv_data_nbx_callback_t)am_data_recv_callback;
  if (memh) {
    param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
    param.memh = memh;
  }

  ucs_status_ptr_t sptr =
    ucp_am_recv_data_nbx(worker_, data_desc.desc, buffer, length, &param);
  return process_request(
    "ucp_am_recv_data_nbx", sptr, std::move(callback), UcxRequestType::Recv);
}

void UcxConnection::cancel_request(UcxRequest* request) {
  UCX_CONN_DEBUG << "canceling " << request->what << " request " << request
                 << "\n";
  ucp_request_cancel(worker_, request);
}

void UcxConnection::cancel_send() {
  if (ucs_list_is_empty(&all_requests_)) {
    return;
  }

  UcxRequest *request, *tmp;
  unsigned count = 0;
  ucs_list_for_each_safe(request, tmp, &all_requests_, pos) {
    ++count;
    if (request->type == UcxRequestType::Send) {
      UCX_CONN_DEBUG << "canceling " << request->what << " request " << request
                     << " #" << count << "\n";
      ucp_request_cancel(worker_, request);
    }
  }
}

void UcxConnection::cancel_recv() {
  if (ucs_list_is_empty(&all_requests_)) {
    return;
  }

  UcxRequest *request, *tmp;
  unsigned count = 0;
  ucs_list_for_each_safe(request, tmp, &all_requests_, pos) {
    ++count;
    if (request->type == UcxRequestType::Recv) {
      UCX_CONN_DEBUG << "canceling " << request->what << " request " << request
                     << " #" << count << "\n";
      ucp_request_cancel(worker_, request);
    }
  }
}

void UcxConnection::cancel_all() {
  if (ucs_list_is_empty(&all_requests_)) {
    return;
  }

  UcxRequest *request, *tmp;
  unsigned count = 0;
  ucs_list_for_each_safe(request, tmp, &all_requests_, pos) {
    ++count;
    UCX_CONN_DEBUG << "canceling " << request->what << " request " << request
                   << " #" << count;
    ucp_request_cancel(worker_, request);
  }
}

void UcxConnection::handle_connection_error(ucs_status_t status) {
  if (UCS_STATUS_IS_ERR(ucx_status_) || is_disconnecting()) {
    return;
  }

  UCX_CONN_LOG << "detected error: " << ucs_status_string(status) << "\n";
  ucx_status_ = status;

  /* the upper layer should close the connection */
  if (is_established()) {
    handle_err_cb_->handle_connection_error(status, id());
  } else {
    invoke_callback(establish_cb_, status);
  }
}

// Private methods implementation
void UcxConnection::common_request_callback(
  void* request, ucs_status_t status) {
  assert(status != UCS_INPROGRESS);

  UcxRequest* r = reinterpret_cast<UcxRequest*>(request);
  assert(r->status == UCS_INPROGRESS);

  r->status = status;
  if (r->callback) {
    // already processed by send/recv function
    (*r->callback)(status);
    if (r->conn.has_value()) {
      r->conn.value().get().request_completed(r);
    }
    request_release(r);
  }
}

void UcxConnection::am_data_recv_callback(
  void* request, ucs_status_t status, size_t length, void* user_data) {
  common_request_callback(request, status);
}

void UcxConnection::error_callback(
  void* arg, ucp_ep_h ep, ucs_status_t status) {
  reinterpret_cast<UcxConnection*>(arg)->handle_connection_error(status);
}

void UcxConnection::set_log_prefix(
  const struct sockaddr* saddr, socklen_t addrlen) {
  std::stringstream ss;
  remote_address_ = sockaddr_str(saddr, addrlen);
  ss << "[UCX-connection " << this << ": #" << conn_id_ << " "
     << remote_address_ << "]";
  memset(log_prefix_, 0, MAX_LOG_PREFIX_SIZE);
  auto length = ss.str().length();
  if (length >= MAX_LOG_PREFIX_SIZE) {
    length = MAX_LOG_PREFIX_SIZE - 1;
  }
  memcpy(log_prefix_, ss.str().c_str(), length);
}

void UcxConnection::connect_common(
  ucp_ep_params_t& ep_params, std::unique_ptr<UcxCallback> callback) {
  establish_cb_ = std::move(callback);

  // create endpoint
  ep_params.field_mask |=
    UCP_EP_PARAM_FIELD_ERR_HANDLER | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
  ep_params.err_handler.cb = error_callback;
  ep_params.err_handler.arg = reinterpret_cast<void*>(this);

  ucs_status_t status = ucp_ep_create(worker_, &ep_params, &ep_);
  if (status != UCS_OK) {
    assert(ep_ == nullptr);
    UCX_CONN_ERROR << "ucp_ep_create() failed: " << ucs_status_string(status)
                   << "\n";
    handle_connection_error(status);
    return;
  }

  UCX_CONN_DEBUG << "created endpoint " << ep_ << ", connection id " << conn_id_
                 << "\n";

  // When connecting sucessfully, the establish_cb_ will be called.
  // establish_cb_ will call on_complete function and then add this connection
  // into context
  connect_am(std::move(callback));
}

void UcxConnection::connect_am(std::unique_ptr<UcxCallback> callback) {
  // With AM use ep as a connection ID. AM receive callback provides
  // reply ep, which can be used for finding a proper connection.
  conn_id_ = reinterpret_cast<uint64_t>(ep_);
  established(UCS_OK);
}

void UcxConnection::established(ucs_status_t status) {
  ucx_status_ = status;
  invoke_callback(establish_cb_, status);
}

void UcxConnection::request_started(UcxRequest* r) {
  ucs_list_add_tail(&all_requests_, &r->pos);
}

void UcxConnection::request_completed(UcxRequest* r) {
  const auto conn_ref = r->conn.value();
  assert(&(conn_ref.get()) == this);
  ucs_list_del(&r->pos);

  if (disconnect_cb_ != nullptr) {
    UCX_CONN_ERROR << "completing " << r->what << " request " << r
                   << " with status \"" << ucs_status_string(r->status)
                   << "\" (" << r->status << ")"
                   << " during disconnect"
                   << "\n";
  }
}

void UcxConnection::ep_close(enum ucp_ep_close_mode mode) {
  static constexpr const std::array<const std::string_view, 2> mode_str = {
    "force", "flush"};
  if (ep_ == nullptr) {
    return;
  }

  assert(close_request_ == nullptr);

  UCX_CONN_DEBUG << "closing ep " << ep_ << " mode " << mode_str[mode] << "\n";
  close_request_ = ucp_ep_close_nb(ep_, mode);
  ep_ = nullptr;
}

std::tuple<ucs_status_t, UcxRequest*> UcxConnection::process_request(
  std::string_view what, ucs_status_ptr_t ptr_status,
  std::unique_ptr<UcxCallback> callback, UcxRequestType type) {
  ucs_status_t status = UCS_OK;

  if (ptr_status == nullptr) {
    (*callback)(UCS_OK);
    return std::make_tuple(UCS_OK, nullptr);
  } else if (UCS_PTR_IS_ERR(ptr_status)) {
    status = UCS_PTR_STATUS(ptr_status);
    UCX_CONN_ERROR << what
                   << " failed with status: " << ucs_status_string(status)
                   << "\n";
    (*callback)(status);
    return std::make_tuple(status, nullptr);
  } else {
    UcxRequest* r = reinterpret_cast<UcxRequest*>(ptr_status);
    if (r->status != UCS_INPROGRESS) {
      assert(ucp_request_is_completed(r));
      status = r->status;
      (*callback)(status);
      request_release(r);
      r = nullptr;
    } else {
      r->callback = std::move(callback);
      r->conn = std::ref(*this);
      r->type = type;
      r->what = what;
      request_started(r);
    }
    return std::make_tuple(status, r);
  }

  return std::make_tuple(UCS_ERR_LAST, nullptr);
}

void UcxConnection::invoke_callback(
  std::unique_ptr<UcxCallback>& callback, ucs_status_t status) {
  if (callback != nullptr) {
    auto cb = std::move(callback);
    (*cb)(status);
  }
}

void UcxConnection::request_init(void* request) {
  UcxRequest* r = reinterpret_cast<UcxRequest*>(request);
  request_reset(r);
}

void UcxConnection::request_reset(UcxRequest* r) {
  if (
    r->callback || r->callback.get() != nullptr
    || r->callback.get()
         != reinterpret_cast<UcxCallback*>(
           sizeof(size_t) == 8 ? 0xffffffffffffffff : 0xffffffff)) {
    r->callback.release();
  }
  r->callback = nullptr;
  r->conn = std::nullopt;
  r->status = UCS_INPROGRESS;
  r->type = UcxRequestType::Unknown;
  r->pos.next = nullptr;
  r->pos.prev = nullptr;
}

void UcxConnection::request_release(void* request) {
  request_reset(reinterpret_cast<UcxRequest*>(request));
  ucp_request_free(request);
}

const std::string UcxConnection::sockaddr_str(
  const struct sockaddr* saddr, size_t addrlen) {
  char buf[128];
  uint16_t port;

  if (saddr->sa_family != AF_INET) {
    return "<unknown address family>";
  }

  switch (saddr->sa_family) {
    case AF_INET: {
      const struct sockaddr_in* sa_in = (const struct sockaddr_in*)saddr;
      inet_ntop(AF_INET, &sa_in->sin_addr, buf, sizeof(buf));
      port = ntohs(sa_in->sin_port);
      break;
    }
    case AF_INET6: {
      const struct sockaddr_in6* sa_in6 = (const struct sockaddr_in6*)saddr;
      inet_ntop(AF_INET6, &sa_in6->sin6_addr, buf, sizeof(buf));
      port = ntohs(sa_in6->sin6_port);
      break;
    }
    default:
      return "<invalid address>";
  }

  snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ":%u", port);
  return buf;
}

}  // namespace stdexe_ucx_runtime
