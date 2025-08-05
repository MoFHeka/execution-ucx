/*Copyright (c) Facebook, Inc. and its affiliates.
Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

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

#include "ucx_context/ucx_am_context/ucx_am_context.hpp"

#include <arpa/inet.h>
#include <signal.h>
#include <time.h>
#include <ucp/api/ucp.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ucx_context/ucx_connection.hpp"
#include "ucx_context/ucx_context_logger.hpp"

namespace stdexe_ucx_runtime {

static thread_local ucx_am_context* currentThreadContext;

static constexpr uint64_t remote_queue_event_user_data = 0;

// #define LOGGING_ENABLED

#ifdef LOGGING_ENABLED
#define LOG(S)             \
  do {                     \
    ::std::puts(S);        \
    ::std::fflush(stdout); \
  } while (false)
#define LOGX(...)               \
  do {                          \
    ::std::printf(__VA_ARGS__); \
    ::std::fflush(stdout);      \
  } while (false)
#else
#define LOG(S) \
  do {         \
  } while (false)
#define LOGX(...) \
  do {            \
  } while (false)
#endif

////////////////////////////////////////////////////////////
// UCX AM Context structures
//
//

ucx_am_context::ucx_am_context(
  const std::unique_ptr<ucx_memory_resource>& memoryResource,
  const std::string_view ucxContextName,
  const time_duration connectionTimeout,
  const std::optional<std::reference_wrapper<UcxAutoDeviceContext>>
    deviceContext,
  const uint64_t clientId)
  : mr_(memoryResource),
    ucxAmContextName_(ucxContextName),
    connTimeout_(connectionTimeout),
    deviceContext_(deviceContext),
    clientId_(clientId) {
  std::error_code ec = ucx_am_context::init(ucxContextName);
  if (ec) {
    throw std::runtime_error(
      "UCX Context initialization failed: " + ec.message());
  }

  set_setimer_action_event();

  ucpContextInitialized_ = true;
  UCX_CTX_DEBUG << "UCX Context construction done\n";
}

ucx_am_context::ucx_am_context(
  const std::unique_ptr<ucx_memory_resource>& memoryResource,
  const ucp_context_h ucpContext,
  const time_duration connectionTimeout,
  const std::optional<std::reference_wrapper<UcxAutoDeviceContext>>
    deviceContext,
  const uint64_t clientId)
  : mr_(memoryResource),
    connTimeout_(connectionTimeout),
    deviceContext_(deviceContext),
    clientId_(clientId),
    ucpContext_(const_cast<ucp_context_h>(ucpContext)) {
  assert(ucpContext != nullptr);
  isUcpContextExternal_ = true;
  ucp_context_attr_t ucpCtxAttr;
  ucs_status_t status = ucp_context_query(ucpContext, &ucpCtxAttr);
  if (status != UCS_OK) {
    throw std::runtime_error(
      "UCX Context query failed from passed ucp_context_h");
  }
  ucxAmContextName_ = std::string(ucpCtxAttr.name);

  std::error_code ec = ucx_am_context::init_with_internal_ucp_context();
  if (ec) {
    throw std::runtime_error(
      "UCX Context initialization failed: " + ec.message());
  }

  set_setimer_action_event();

  ucpContextInitialized_ = true;
  UCX_CTX_DEBUG << "UCX Context construction done\n";
}

ucx_am_context::~ucx_am_context() {
  destroy_connections();

  if (ucpContext_ && !isUcpContextExternal_) {
    ucx_am_context::destroy_ucp_context(ucpContext_);
  }
}

void ucx_am_context::run_impl(const bool& shouldStop) {
  LOG("run loop started");

  auto* oldContext = std::exchange(currentThreadContext, this);
  scope_guard g = [=]() noexcept {
    std::exchange(currentThreadContext, oldContext);
    LOG("run loop exited");
  };

  if (ucpContextInitialized_) {
    [[maybe_unused]] unsigned result = progress_worker_event();
    LOGX("ucp_worker_progress() returned - finished %u callbacks\n", result);
  }

  // Activate the device context if it is provided.
  std::unique_ptr<OperationRAII> deviceAutoContextOperation = nullptr;
  if (deviceContext_.has_value()) {
    deviceAutoContextOperation =
      (deviceContext_.value().get())(ucpContext_, ucpWorker_);
  }

  while (ucpContextInitialized_) {
    // Dequeue and process local queue items (ready to run)
    execute_pending_local();

    if (shouldStop) {
      break;
    }

    // Check for any new completion-queue items.
    acquire_completion_queue_items();

    if (timersAreDirty_) {
      update_timers();
    }

    LOGX("sqUnflushedCount_ %u\n", sqUnflushedCount_);

    // Check for remotely-queued items.
    // Only do this if we haven't submitted a poll operation for the
    // completion queue - in which case we'll just wait until we receive the
    // completion-queue item).
    if (!remoteQueueReadSubmitted_) {
      acquire_remote_queued_items();
    }

    // Process additional I/O requests that were waiting for
    // additional space either in the submission queue or the completion queue.
    while (!pendingIoQueue_.empty() && can_submit_io()) {
      auto* item = pendingIoQueue_.pop_front();
      item->execute_(item);
    }

    if (localQueue_.empty() || sqUnflushedCount_ > 0) {
      const bool isIdle = sqUnflushedCount_ == 0 && localQueue_.empty();
      if (isIdle) {
        if (!remoteQueueReadSubmitted_) {
          LOG("try_register_remote_queue_notification()");
          remoteQueueReadSubmitted_ = try_register_remote_queue_notification();
        }
      }

      LOGX(
        "ucx_am_context::run_impl() - submit %u, pending %u\n",
        sqUnflushedCount_, pending_operation_count());

      [[maybe_unused]] unsigned result = progress_worker_event();

      LOGX("ucp_worker_progress() returned - finished %u callbacks\n", result);
    }
  }
}

bool ucx_am_context::is_running_on_io_thread() const noexcept {
  return this == currentThreadContext;
}

void ucx_am_context::schedule_impl(operation_base* op) {
  UNIFEX_ASSERT(op != nullptr);
  if (is_running_on_io_thread()) {
    schedule_local(op);
  } else {
    schedule_remote(op);
  }
}

void ucx_am_context::schedule_local(operation_base* op) noexcept {
  localQueue_.push_back(op);
}

void ucx_am_context::schedule_local(operation_queue ops) noexcept {
  localQueue_.append(std::move(ops));
}

void ucx_am_context::schedule_remote(operation_base* op) noexcept {
  bool ioThreadWasInactive = remoteQueue_.enqueue(op);

  if (ioThreadWasInactive) {
    // We were the first to queue an item and the I/O thread is not
    // going to check the queue until we signal it that new items
    // have been enqueued remotely.
    signal_remote_queue();
  }
}

void ucx_am_context::schedule_pending_io(operation_base* op) noexcept {
  UNIFEX_ASSERT(is_running_on_io_thread());
  pendingIoQueue_.push_back(op);
}

void ucx_am_context::reschedule_pending_io(operation_base* op) noexcept {
  UNIFEX_ASSERT(is_running_on_io_thread());
  pendingIoQueue_.push_front(op);
}

void ucx_am_context::schedule_at_impl(schedule_at_operation* op) noexcept {
  UNIFEX_ASSERT(is_running_on_io_thread());
  timers_.insert(op);
  if (timers_.top() == op) {
    timersAreDirty_ = true;
  }
}

void ucx_am_context::execute_pending_local() noexcept {
  if (localQueue_.empty()) {
    LOG("local queue is empty");
    return;
  }

  LOG("processing local queue items");

  [[maybe_unused]] size_t count = 0;
  auto pending = std::move(localQueue_);
  while (!pending.empty()) {
    auto* item = pending.pop_front();
    item->execute_(item);
    ++count;
  }

  LOGX("processed %zu local queue items\n", count);
}

void ucx_am_context::acquire_completion_queue_items() noexcept {
  // Use 'relaxed' load for the head since it should only ever
  // be modified by the current thread.
  std::uint32_t cqHead = cqHead_.load(std::memory_order_relaxed);
  std::uint32_t cqTail = cqTail_.load(std::memory_order_acquire);
  LOGX("completion queue head = %u, tail = %u\n", cqHead, cqTail);

  if (cqHead != cqTail) {
    const auto mask = cqMask_;
    const auto count = cqTail - cqHead;
    UNIFEX_ASSERT(count <= static_cast<std::uint32_t>(cqEntryCount_));

    operation_base head;

    LOGX("got %u completions\n", count);

    operation_queue completionQueue;

    for (std::uint32_t i = 0; i < count; ++i) {
      auto& cqe = cqEntries_[(cqHead + i) & mask];

      if (cqe.user_data == remote_queue_event_user_data) {
        LOG("got remote queue wakeup");

        // Skip processing this item and let the loop check
        // for the remote-queued items next time around.
        remoteQueueReadSubmitted_ = false;
        continue;
      } else if (cqe.user_data == timer_user_data()) {
        LOGX("got timer completion result %i\n", cqe.res);
        UNIFEX_ASSERT(activeTimerCount_ > 0);
        --activeTimerCount_;

        LOGX("now %u active timers\n", activeTimerCount_);
        if (cqe.res != ECANCELED) {
          LOG("timer not cancelled, marking timers as dirty");
          timersAreDirty_ = true;
        }

        if (activeTimerCount_ == 0) {
          LOG("no more timers, resetting current due time");
          currentDueTime_.reset();
        }
        continue;
      } else if (cqe.user_data == remove_timer_user_data()) {
        // Ignore timer cancellation completion.
        continue;
      }

      auto& completionState = *reinterpret_cast<completion_base*>(
        static_cast<std::uintptr_t>(cqe.user_data));

      // Save the result in the completion state.
      completionState.result_ = cqe.res;

      // Add it to a temporary queue of newly completed items.
      completionQueue.push_back(&completionState);
    }

    schedule_local(std::move(completionQueue));

    // Mark those completion queue entries as consumed.
    cqHead_.store(cqTail, std::memory_order_release);

    cqPendingCount_ -= count;
  }
}

void ucx_am_context::acquire_remote_queued_items() noexcept {
  UNIFEX_ASSERT(!remoteQueueReadSubmitted_);
  auto items = remoteQueue_.dequeue_all();
  LOG(
    items.empty() ? "remote queue is empty"
                  : "acquired items from remote queue");
  schedule_local(std::move(items));
}

bool ucx_am_context::try_register_remote_queue_notification() noexcept {
  // Check that we haven't already hit the limit of pending
  // I/O completion events.
  const auto populateRemoteQueueSqe = [this]() noexcept {
    auto queuedItems = remoteQueue_.try_mark_inactive_or_dequeue_all();
    if (!queuedItems.empty()) {
      schedule_local(std::move(queuedItems));
      return false;
    }

    remoteQueueEventEntry_.store(true, std::memory_order_release);
    // Waiting for remoteQueueEventEntry_ to be set by the I/O thread
    return true;
  };

  if (try_submit_io(populateRemoteQueueSqe)) {
    LOG("added remote queue to submission queue");
    return true;
  }

  return false;
}

void ucx_am_context::signal_remote_queue() {
  LOG("notifying remote queue");

  if (remoteQueueEventEntry_.load(std::memory_order_acquire)) {
    auto& entry = get_completion_queue_entry();
    entry.user_data = remote_queue_event_user_data;
    entry.res = 0;

    // Reset remoteQueueEventEntry_.
    remoteQueueEventEntry_.store(false, std::memory_order_relaxed);
  }
}

void ucx_am_context::remove_timer(schedule_at_operation* op) noexcept {
  LOGX("remove_timer(%p)\n", (void*)op);

  UNIFEX_ASSERT(!timers_.empty());
  if (timers_.top() == op) {
    timersAreDirty_ = true;
  }
  timers_.remove(op);
}

void ucx_am_context::update_timers() noexcept {
  LOG("update_timers()");

  // Reap any elapsed timers.
  if (!timers_.empty()) {
    time_point now = monotonic_clock::now();
    while (!timers_.empty() && timers_.top()->dueTime_ <= now) {
      schedule_at_operation* item = timers_.pop();

      LOGX("dequeued elapsed timer %p\n", (void*)item);

      if (item->canBeCancelled_) {
        auto oldState = item->state_.fetch_add(
          schedule_at_operation::timer_elapsed_flag, std::memory_order_acq_rel);
        if ((oldState & schedule_at_operation::cancel_pending_flag) != 0) {
          LOGX("timer already cancelled\n");

          // Timer has been cancelled by a remote thread.
          // The other thread is responsible for enqueueing is operation onto
          // the remoteQueue_.
          continue;
        }
      }

      // Otherwise, we are responsible for enqueuing the timer onto the
      // ready-to-run queue.
      schedule_local(item);
    }
  }

  // Check if we need to cancel or start some new OS timers.
  if (timers_.empty()) {
    if (currentDueTime_.has_value()) {
      LOG("no more schedule_at requests, cancelling timer");

      // Cancel the outstanding timer.
      if (try_submit_timer_io_cancel()) {
        currentDueTime_.reset();
        timersAreDirty_ = false;
      }
    }
  } else {
    const auto earliestDueTime = timers_.top()->dueTime_;

    if (currentDueTime_) {
      constexpr auto threshold = std::chrono::microseconds(1);
      if (earliestDueTime < (*currentDueTime_ - threshold)) {
        LOG("active timer, need to cancel and submit an earlier one");

        // An earlier time has been scheduled.
        // Cancel the old timer before submitting a new one.
        if (try_submit_timer_io_cancel()) {
          currentDueTime_.reset();
          if (try_submit_timer_io(earliestDueTime)) {
            currentDueTime_ = earliestDueTime;
            timersAreDirty_ = false;
          }
        }
      } else {
        timersAreDirty_ = false;
      }
    } else {
      // No active timer, submit a new timer
      LOG("no active timer, trying to submit a new one");
      if (try_submit_timer_io(earliestDueTime)) {
        currentDueTime_ = earliestDueTime;
        timersAreDirty_ = false;
      }
    }
  }
}

/*
// BACKUP(He Jia): use pthread signal to trigger timer timeout
void ucx_am_context::timer_timeout_callback(union sigval sv) noexcept {
  ucx_am_context* self = static_cast<ucx_am_context*>(sv.sival_ptr);

  auto& entry = self->get_completion_queue_entry();
  entry.user_data = self->timer_user_data();
  entry.res = 0;

  UNIFEX_ASSERT(self->timers_.top());
  (self->timers_.top())->cq_entry_ref_ = std::ref(entry);
}
*/

void ucx_am_context::timer_timeout_callback(
  int signo, siginfo_t* info, void* _) {
  if (signo == SIGUSR2) {
    ucx_am_context* self =
      static_cast<ucx_am_context*>(info->si_value.sival_ptr);

    auto& entry = self->get_completion_queue_entry();
    entry.user_data = self->timer_user_data();
    entry.res = 0;

    UNIFEX_ASSERT(self->timers_.top());
    (self->timers_.top())->cq_entry_ref_ = std::ref(entry);
  }
}

bool ucx_am_context::try_submit_timer_io(const time_point& dueTime) noexcept {
  auto populateSqe = [&]() noexcept {
    auto fail_fn = [this]() {
      auto& entry = this->get_completion_queue_entry();
      entry.user_data = timer_user_data();
      entry.res = ECANCELED;
    };

    auto succ_fn = [this]() {
      auto& entry = this->get_completion_queue_entry();
      entry.user_data = timer_user_data();
      entry.res = 0;

      lastTimerOp_->cq_entry_ref_ = std::ref(entry);
    };

    lastTimerOp_ = timers_.top();

    /*
    // BACKUP(He Jia): use pthread signal to trigger timer timeout
    // // Set pthread stack size to 8MB
    // pthread_attr_t attr;
    // pthread_attr_init(&attr);
    // pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
    // pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    // pthread_attr_destroy(&attr);

    timerSigevent_.sigev_notify = SIGEV_THREAD;
    timerSigevent_.sigev_notify_function = timer_timeout_callback;
    timerSigevent_.sigev_value.sival_ptr = this;
    */

    auto dur_ = dueTime - monotonic_clock::now();
    auto sec_ = std::chrono::duration_cast<std::chrono::seconds>(dur_).count();
    auto nsec_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   dur_ - std::chrono::seconds(sec_))
                   .count();
    if (sec_ < 0 || nsec_ <= 0) {
      return succ_fn();
    }
    timerItimerspec_.it_value.tv_sec = sec_;
    timerItimerspec_.it_value.tv_nsec = nsec_;
    timerItimerspec_.it_interval.tv_sec = 0;
    timerItimerspec_.it_interval.tv_nsec = 0;

    if (timer_settime(timerId_, 0, &timerItimerspec_, NULL) != 0) {
      return fail_fn();
    }
  };

  if (try_submit_io(populateSqe)) {
    ++activeTimerCount_;
    return true;
  }

  return false;
}

bool ucx_am_context::try_submit_timer_io_cancel() noexcept {
  auto populateSqe = [&]() noexcept {
    auto& entry = this->get_completion_queue_entry();
    entry.user_data = remove_timer_user_data();
    entry.res = 0;

    if (timerId_) {
      timer_delete(timerId_);
    }

    if (lastTimerOp_ != nullptr) {
      if (lastTimerOp_->cq_entry_ref_.has_value()) {
        lastTimerOp_->cq_entry_ref_.value().get().res = ECANCELED;
      }
      lastTimerOp_ = nullptr;
    } else {
      LOG("no last timer operation, skipping cancel");
    }
  };

  return try_submit_io(populateSqe);
}

std::error_code ucx_am_context::init_ucp_context(
  std::string_view name, ucp_context_h& ucpContext, bool printConfig) {
  ucs_status_t status;

  ucp_params_t ucp_params;
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES
                          | UCP_PARAM_FIELD_REQUEST_INIT
                          | UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_NAME;
  ucp_params.features = UCP_FEATURE_AM;
  ucp_params.request_init = ucx_connection::request_init;
  ucp_params.request_size = sizeof(ucx_request);
  ucp_params.name = name.data();

  ucp_config_t* config;
  status = ucp_config_read(nullptr, nullptr, &config);
  assert(status == UCS_OK);

  if (printConfig) {
    ucp_config_print(config, stdout, NULL, UCS_CONFIG_PRINT_CONFIG);
  }

  status = ucp_init(&ucp_params, config, &ucpContext);

  ucp_config_release(config);

  return stdexe_ucx_runtime::make_error_code(status);
}

void ucx_am_context::destroy_ucp_context(ucp_context_h ucpContext) {
  if (ucpContext) {
    ucp_cleanup(ucpContext);
  }
}

bool ucx_am_context::is_ucp_context_external() const noexcept {
  return isUcpContextExternal_;
}

std::error_code ucx_am_context::init_ucp_worker() {
  ucp_worker_params_t worker_params;
  worker_params.field_mask =
    UCP_WORKER_PARAM_FIELD_THREAD_MODE | UCP_WORKER_PARAM_FIELD_CLIENT_ID;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  worker_params.client_id = clientId_;

  ucs_status_t status =
    ucp_worker_create(ucpContext_, &worker_params, &ucpWorker_);
  return stdexe_ucx_runtime::make_error_code(status);
}

void ucx_am_context::set_setimer_action_event() {
  timerSigaction_.sa_sigaction = timer_timeout_callback;
  timerSigaction_.sa_flags = SA_SIGINFO;
  sigaction(SIGUSR2, &timerSigaction_, NULL);

  timerSigevent_.sigev_notify = SIGEV_SIGNAL;
  timerSigevent_.sigev_signo = SIGUSR2;
  timerSigevent_.sigev_value.sival_ptr = this;

  if (timer_create(CLOCK_REALTIME, &timerSigevent_, &timerId_) != 0) {
    throw std::runtime_error("UCX Context timer creation failed");
  }
}

std::error_code ucx_am_context::init_with_internal_ucp_context(bool forceInit) {
  std::error_code ec;

  /* Create worker */
  ec = init_ucp_worker();
  if (ec) {
    ucp_cleanup(ucpContext_);
    ucpContext_ = nullptr;
    UCX_CTX_ERROR << "ucp_worker_create() failed: " << ec.message() << "\n";
    return ec;
  }

  LOGX("created worker %p\n", ucpWorker_);

  set_am_handler(am_recv_callback, this);

  ucpContextInitialized_ = true;
  return ec;
}

std::error_code ucx_am_context::init(std::string_view name, bool forceInit) {
  std::error_code ec;

  if (ucpContextInitialized_ && !forceInit) {
    UCX_CTX_INFO << "UCP Context already initialized\n";
    return stdexe_ucx_runtime::make_error_code(UCS_OK);
  }

  /* Create context */
  ec = init_ucp_context(name, ucpContext_, forceInit);
  if (ec) {
    UCX_CTX_ERROR << "ucp_init() failed: " << ec.message();
    return ec;
  }

  LOGX("created context %p with AM \n", ucpContext_);

  return init_with_internal_ucp_context(forceInit);
}

std::uint64_t ucx_am_context::get_client_id() noexcept {
  return conn_id_it_.fetch_add(1, std::memory_order_relaxed);
}

void ucx_am_context::set_am_handler(ucp_am_recv_callback_t cb, void* arg) {
  ucp_am_handler_param_t param;

  param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID
                     | UCP_AM_HANDLER_PARAM_FIELD_CB
                     | UCP_AM_HANDLER_PARAM_FIELD_ARG;
  /*
  // BACKUP(He Jia): use UCP_AM_RECV_ATTR_FLAG_DATA to indicate eager message
  // That means handle data body outside of the callback
  param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID
                     | UCP_AM_HANDLER_PARAM_FIELD_CB
                     | UCP_AM_HANDLER_PARAM_FIELD_ARG
                     | UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
  param.flags = UCP_AM_RECV_ATTR_FLAG_DATA;
  */
  param.id = DEFAULT_AM_MSG_ID;
  param.cb = cb;
  param.arg = arg;
  ucp_worker_set_am_recv_handler(ucpWorker_, &param);
}

ucs_status_t ucx_am_context::am_recv_callback(
  void* arg, const void* header, size_t header_length, void* data,
  size_t data_length, const ucp_am_recv_param_t* param) {
  ucx_am_context* self = reinterpret_cast<ucx_am_context*>(arg);
  ucs_status_t status = UCS_OK;

  assert(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP);

  uint64_t conn_id = reinterpret_cast<uint64_t>(param->reply_ep);
  auto conn_opt = self->conn_manager_.get_connection(conn_id);
  if (!conn_opt.has_value()) {
    // TODO(He Jia): change this to assert when AM data dropping is
    // implemented
    UCX_CTX_WARN << "could not find connection with ep " << param->reply_ep
                 << "(" << conn_id << ")\n";
    return status;
  }

  auto conn = conn_opt.value();

  if (conn.get().ucx_status() != UCS_OK) {
    return conn.get().ucx_status();
  }

  // Allocate and copy header in host memory before it is freed
  // TODO(He Jia): Make it more efficient and safe
  void* new_header = self->mr_->allocate(ucx_memory_type::HOST, header_length);
  self->mr_->memcpy(
    ucx_memory_type::HOST, new_header, ucx_memory_type::HOST, header,
    header_length);

  if (
    !(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA)
    && !(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
    void* new_data = self->mr_->allocate(ucx_memory_type::HOST, data_length);
    self->mr_->memcpy(
      ucx_memory_type::HOST, new_data, ucx_memory_type::HOST, data,
      data_length);
    data = new_data;
  } else {
    status = UCS_INPROGRESS;
  }

  // Here data is actually a pointer to the data descriptor, maybe not the
  // real sent data itself
  self->amDescQueue_.emplace_back(
    new_header, header_length, data, data_length, conn_id,
    static_cast<ucp_am_recv_attr_t>(param->recv_attr));

  std::atomic_thread_fence(std::memory_order_acquire);
  if (!self->pendingRecvIoQueue_.empty()) {
    self->reschedule_pending_io(self->pendingRecvIoQueue_.front());
    self->pendingRecvIoQueue_.pop_front();
  }
  std::atomic_thread_fence(std::memory_order_release);

  return status;
}

ucx_am_context::time_duration ucx_am_context::get_conn_timeout()
  const noexcept {
  return connTimeout_;
}

bool ucx_am_context::is_timeout_elapsed(
  time_point timestamp, time_duration timeout) const noexcept {
  return (monotonic_clock::now() - timestamp) > timeout;
}

ucx_am_cqe& ucx_am_context::get_and_update_cq_entry() noexcept {
  const auto tail = cqTail_.load(std::memory_order_relaxed);
  const auto index = tail & cqMask_;
  auto& cqe = cqEntries_.at(index);
  // TODO(He Jia): cqTail_ should be updated in the caller after changing
  // entry(FUCK)
  cqTail_.store(tail + 1, std::memory_order_release);
  return cqe;
}

////////////////////////////////////////////////////////////
// UCX Context Listener

ucs_status_t ucx_am_context::listen(
  const std::unique_ptr<sockaddr>& socket, size_t addrlen) {
  auto addr = reinterpret_cast<const struct sockaddr*>(socket.get());
  ucp_listener_params_t listener_params;

  listener_params.field_mask =
    UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  listener_params.sockaddr.addr = addr;
  listener_params.sockaddr.addrlen = addrlen;
  listener_params.conn_handler.cb = connect_request_callback;
  listener_params.conn_handler.arg = reinterpret_cast<void*>(this);

  ucs_status_t status =
    ucp_listener_create(ucpWorker_, &listener_params, &ucpListener_);
  if (status != UCS_OK) {
    UCX_CTX_ERROR << "ucp_listener_create() failed: "
                  << ucs_status_string(status);
    return status;
  }

  UCX_CTX_TRACE << "started listener on "
                << ucx_connection::sockaddr_str(addr, addrlen) << "\n";
  return UCS_OK;
}

ucs_status_t ucx_am_context::is_listener_valid() const noexcept {
  if (!ucpListener_) {
    return UCS_ERR_UNREACHABLE;
  }
  ucp_listener_attr_t attr;
  /* Query the created listener to get the port it is listening on. */
  attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
  auto status = ucp_listener_query(ucpListener_, &attr);
  if (status != UCS_OK) {
    UCX_CTX_ERROR << "ucp_listener_query() failed: "
                  << ucs_status_string(status);
  }
  return status;
}

void ucx_am_context::destroy_listener() {
  if (ucpListener_) {
    ucp_listener_destroy(ucpListener_);
    ucpListener_ = nullptr;
  }
}

// Create a new connection

std::uint64_t ucx_am_context::create_new_connection(
  const struct sockaddr* src_saddr, const struct sockaddr* dst_saddr,
  socklen_t addrlen) {
  // Create a new connection instance
  auto new_conn = std::make_unique<ucx_connection>(
    ucpWorker_, std::make_unique<ucx_handle_err_callback>(*this),
    [this]() { return get_client_id(); });

  // Call UcxConnection::connect to establish connection
  new_conn->connect(
    src_saddr, dst_saddr, addrlen,
    std::make_unique<ucx_connect_callback>(*this, new_conn.get()));

  // Get new connection ID and add to active map
  auto conn_id = new_conn->id();
  if (new_conn->ucx_status() == UCS_OK) {
    conn_manager_.add_connection(conn_id, std::move(new_conn));
  } else {
    new_conn->disconnect(std::make_unique<ucx_disconnect_callback>(*this));
  }

  return conn_id;
}

std::uint64_t ucx_am_context::recreate_connection_from_failed(
  conn_pair_t conn) {
  // Warning: old connection should be removed outside of this function
  auto& failed_conn = conn.second;

  // Parse address information from failed connection
  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;

  // Parse IP and port from remote_address_
  const std::string& peer_name = failed_conn.get().get_peer_name();
  if (!peer_name.empty()) {
    size_t pos = peer_name.find(':');
    if (pos != std::string::npos) {
      // Extract IP and port
      std::string ip = peer_name.substr(0, pos);
      std::string port = peer_name.substr(pos + 1);

      // Convert IP string to network address
      inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
      // Convert port string to network byte order
      addr.sin_port = htons(std::stoi(port));

      // Create new connection using the parsed address
      return create_new_connection(NULL, (struct sockaddr*)&addr, sizeof(addr));
    }
  }

  return 0;
}

// UCX Worker memory map
std::error_code ucx_am_context::register_ucx_memory(
  void* ptr, size_t size, ucp_mem_h& memh) {
  ucp_mem_map_params_t mem_map_params;
  mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS
                              | UCP_MEM_MAP_PARAM_FIELD_LENGTH
                              | UCP_MEM_MAP_PARAM_FIELD_FLAGS;
  mem_map_params.address = ptr;
  mem_map_params.length = size;
  mem_map_params.flags = UCP_MEM_MAP_NONBLOCK;

  ucs_status_t status = ucp_mem_map(ucpContext_, &mem_map_params, &memh);
  if (status != UCS_OK) {
    UCX_CTX_ERROR << "ucp_mem_map() failed: " << ucs_status_string(status);
  }
  return stdexe_ucx_runtime::make_error_code(status);
}

void ucx_am_context::unregister_ucx_memory(ucp_mem_h& memh) {
  ucp_mem_unmap(ucpContext_, memh);
}

// UCX Context Progress Function

unsigned ucx_am_context::progress_worker_event() {
  return ucp_worker_progress(ucpWorker_);
}

std::tuple<ucs_status_t, std::uint64_t> ucx_am_context::progress_conn_request(
  const ep_conn_request& epConnReq) {
  ucs_status_t status = UCS_OK;
  std::uint64_t conn_id = std::uintptr_t(nullptr);

  if (is_timeout_elapsed(epConnReq.arrival_time, get_conn_timeout())) {
    UCX_CTX_INFO << "reject connection request " << epConnReq.conn_request
                 << " since server's timeout ("
                 << std::chrono::duration_cast<std::chrono::seconds>(
                      get_conn_timeout())
                      .count()
                 << " seconds) elapsed\n";
    status = ucp_listener_reject(ucpListener_, epConnReq.conn_request);
  } else {
    auto conn = std::make_unique<ucx_connection>(
      ucpWorker_, std::make_unique<ucx_handle_err_callback>(*this),
      [this]() { return get_client_id(); });
    // TODO(He Jia): Fix this ugly code, should not use raw pointer
    conn->accept(
      epConnReq.conn_request,
      std::make_unique<ucx_accept_callback>(*this, conn.get()));
    // TODO(He Jia): This state may not reflect the real situation because
    // it is possible that the establish callback has not been run at this
    // time
    status = conn->ucx_status();
    // In C++17 and earlier, the evaluation order of function arguments is
    // unspecified, which poses a theoretical risk of undefined behavior.
    // Starting from C++20, the evaluation order is from left to right.
    if (status == UCS_OK) {
      conn_id = conn->id();
      conn_manager_.add_connection(conn_id, std::move(conn));
    } else if ((status != UCS_ERR_CANCELED) || !conn->is_disconnecting()) {
      conn->disconnect(std::make_unique<ucx_disconnect_callback>(*this));
    }
  }
  return {status, conn_id};
}

std::vector<std::pair<std::uint64_t, std::error_code>>
ucx_am_context::progress_pending_conn_requests() {
  std::vector<std::pair<std::uint64_t, std::error_code>> conn_id_status_vector;

  while (!epConnReqQueue_.empty()) {
    const auto epConnReq = std::move(epConnReqQueue_.front());
    auto [status, conn_id] = progress_conn_request(epConnReq);
    conn_id_status_vector.push_back(
      {conn_id, stdexe_ucx_runtime::make_error_code(status)});
    epConnReqQueue_.pop_front();
  }

  return conn_id_status_vector;
}

void ucx_am_context::progress_failed_connections() {
  conn_manager_.remove_failed_connections();
}

void ucx_am_context::progress_disconnected_connections() {
  conn_manager_.remove_disconnected_connections();
}

void ucx_am_context::wait_disconnected_connections() {
  auto& disconnecting_conns = conn_manager_.get_disconnecting_connections();
  while (!disconnecting_conns.empty()) {
    ucp_worker_progress(ucpWorker_);
    progress_disconnected_connections();
  }
}

void ucx_am_context::destroy_connections() {
  auto& failed_conns = conn_manager_.get_failed_connections();
  for (auto& failed_conn : failed_conns) {
    failed_conn->disconnect_direct();
  }
  // wait for all failed connections being disconnected, call progress to
  // discover new failed connections if any
  while (!failed_conns.empty()) {
    progress_failed_connections();
    ucp_worker_progress(ucpWorker_);
  }

  // disconnect all connections which are not in-progress
  UCX_CTX_INFO << "destroy all connections\n";
  conn_manager_.remove_active_connections_to_disconnecting_queue();
  auto& disconnecting_conns = conn_manager_.get_disconnecting_connections();
  for (auto& disconnecting_conn : disconnecting_conns) {
    disconnecting_conn->disconnect(
      std::make_unique<ucx_disconnect_callback>(*this));
  }

  // wait for all connections being completely disconnected
  wait_disconnected_connections();
}

void ucx_am_context::destroy_worker() {
  if (!ucpWorker_) {
    return;
  }
  ucp_worker_destroy(ucpWorker_);
  ucpWorker_ = NULL;
}

////////////////////////////////////////////////////////////
// UCX Context Connection Callback

void ucx_am_context::connect_request_callback(
  ucp_conn_request_h conn_req, void* arg) {
  ucx_am_context* self = reinterpret_cast<ucx_am_context*>(arg);
  ucp_conn_request_attr_t conn_req_attr;
  ep_conn_request conn_request;
  conn_req_attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR
                             | UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID;
  ucs_status_t status = ucp_conn_request_query(conn_req, &conn_req_attr);
  if (status == UCS_OK) {
    UCX_CTX_TRACE << "got new connection request " << conn_req
                  << " from client "
                  << ucx_connection::sockaddr_str(
                       (const struct sockaddr*)&conn_req_attr.client_address,
                       sizeof(conn_req_attr.client_address))
                  << " client_id " << conn_req_attr.client_id << "\n";
  } else {
    UCX_CTX_ERROR << "got new connection request " << conn_req
                  << ", ucp_conn_request_query() failed ("
                  << ucs_status_string(status) << ")\n";
  }

  conn_request.conn_request = conn_req;
  conn_request.arrival_time = monotonic_clock::now();

  self->epConnReqQueue_.emplace_back(conn_request);

  std::atomic_thread_fence(std::memory_order_acquire);
  if (!self->pendingAcptIoQueue_.empty()) {
    self->reschedule_pending_io(self->pendingAcptIoQueue_.front());
    self->pendingAcptIoQueue_.pop_front();
  }
  std::atomic_thread_fence(std::memory_order_release);
}

void ucx_am_context::dispatch_connection_accepted(
  std::unique_ptr<ucx_connection> connection) {
  conn_manager_.add_connection(connection->id(), std::move(connection));
}

void ucx_am_context::handle_connection_error(std::uint64_t conn_id) {
  conn_manager_.remove_connection_to_failed_queue(conn_id);
}

void ucx_am_context::ucx_disconnect_callback::operator()(ucs_status_t status) {}

void ucx_am_context::ucx_disconnect_callback::mark_inactive(
  std::uint64_t conn_id) {
  context_.conn_manager_.remove_connection_to_inactive_map(conn_id);
}

void ucx_am_context::ucx_disconnect_callback::mark_disconnecting_from_inactive(
  std::uint64_t conn_id) {
  context_.conn_manager_.move_connection_from_inactive_to_disconnecting_queue(
    conn_id);
}

void ucx_am_context::ucx_disconnect_callback::mark_failed_from_inactive(
  std::uint64_t conn_id) {
  context_.conn_manager_.move_connection_from_inactive_to_failed_queue(conn_id);
}

void ucx_am_context::ucx_handle_err_callback::operator()(ucs_status_t status) {
  throw std::runtime_error(
    "Fatal!! ucx_handle_err_callback::operator() should not be called "
    "directly, "
    "Unexpected error: "
    + std::string(ucs_status_string(status)));
}

void ucx_am_context::ucx_handle_err_callback::handle_connection_error(
  ucs_status_t status, std::uint64_t conn_id) {
  context_.handle_connection_error(conn_id);
}

ucx_am_context::connect_sender tag_invoke(
  tag_t<connect_endpoint>,
  ucx_am_context::scheduler scheduler,
  std::unique_ptr<sockaddr>
    src_saddr,
  std::unique_ptr<sockaddr>
    dst_saddr,
  socklen_t addrlen) {
  return ucx_am_context::connect_sender{
    *scheduler.context_, std::move(src_saddr), std::move(dst_saddr), addrlen};
}

ucx_am_context::connect_sender tag_invoke(
  tag_t<connect_endpoint>,
  ucx_am_context::scheduler scheduler,
  std::nullptr_t src_saddr,
  std::unique_ptr<sockaddr>
    dst_saddr,
  socklen_t addrlen) {
  return ucx_am_context::connect_sender{
    *scheduler.context_, nullptr, std::move(dst_saddr), addrlen};
}

ucx_am_context::connect_sender tag_invoke(
  tag_t<connect_endpoint>,
  ucx_am_context::scheduler scheduler,
  std::unique_ptr<sockaddr>
    src_saddr,
  std::unique_ptr<sockaddr>
    dst_saddr,
  size_t addrlen) {
  return ucx_am_context::connect_sender{
    *scheduler.context_, std::move(src_saddr), std::move(dst_saddr),
    static_cast<socklen_t>(addrlen)};
}

ucx_am_context::connect_sender tag_invoke(
  tag_t<connect_endpoint>,
  ucx_am_context::scheduler scheduler,
  std::nullptr_t src_saddr,
  std::unique_ptr<sockaddr>
    dst_saddr,
  size_t addrlen) {
  return ucx_am_context::connect_sender{
    *scheduler.context_, nullptr, std::move(dst_saddr),
    static_cast<socklen_t>(addrlen)};
}

ucx_am_context::accept_connection tag_invoke(
  tag_t<accept_endpoint>, ucx_am_context::scheduler scheduler,
  std::unique_ptr<sockaddr> socket, size_t addrlen) {
  return ucx_am_context::accept_connection{
    *scheduler.context_, std::move(socket), addrlen};
}

ucx_am_context::accept_connection tag_invoke(
  tag_t<accept_endpoint>, ucx_am_context::scheduler scheduler, port_t port) {
  return ucx_am_context::accept_connection{*scheduler.context_, port};
}

ucx_am_context::dispatch_connection_error_sender tag_invoke(
  tag_t<handle_error_connection>, ucx_am_context::scheduler scheduler,
  std::function<bool(std::uint64_t conn_id, ucs_status_t status)> handler) {
  return ucx_am_context::dispatch_connection_error_sender{
    *scheduler.context_, handler};
}

ucx_am_context::send_sender tag_invoke(
  tag_t<connection_send>, ucx_am_context::scheduler scheduler,
  conn_pair_t& conn, ucx_am_data& data) {
  return ucx_am_context::send_sender{*scheduler.context_, conn, data};
}

ucx_am_context::send_sender tag_invoke(
  tag_t<connection_send>,
  ucx_am_context::scheduler scheduler,
  std::uintptr_t conn_id,
  ucx_am_data& data) {
  return ucx_am_context::send_sender{*scheduler.context_, conn_id, data};
}

ucx_am_context::send_sender tag_invoke(
  tag_t<connection_send>, ucx_am_context::scheduler scheduler,
  UcxConnection& conn, ucx_am_data& data) {
  return ucx_am_context::send_sender{*scheduler.context_, conn, data};
}

ucx_am_context::recv_sender tag_invoke(
  tag_t<connection_recv>,
  ucx_am_context::scheduler scheduler,
  ucx_am_data& data) {
  return ucx_am_context::recv_sender{*scheduler.context_, data};
}

ucx_am_context::recv_sender tag_invoke(
  tag_t<connection_recv>,
  ucx_am_context::scheduler scheduler,
  ucx_memory_type data_type) {
  return ucx_am_context::recv_sender{*scheduler.context_, data_type};
}

}  // namespace stdexe_ucx_runtime
