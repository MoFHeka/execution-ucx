#include <iostream>
#include <unifex/inplace_stop_token.hpp>
#include <unifex/just.hpp>
#include <unifex/just_error.hpp>
#include <unifex/just_from.hpp>
#include <unifex/let_done.hpp>
#include <unifex/let_error.hpp>
#include <unifex/on.hpp>
#include <unifex/single_thread_context.hpp>
#include <unifex/spawn_detached.hpp>
#include <unifex/static_thread_pool.hpp>
#include <unifex/stop_when.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/timed_single_thread_context.hpp>
#include <unifex/upon_done.hpp>
#include <unifex/upon_error.hpp>
#include <unifex/v2/async_scope.hpp>
#include <unifex/variant_sender.hpp>
#include <unifex/when_any.hpp>
#include <unifex/with_query_value.hpp>

using namespace std;
using namespace unifex;

void print_error(std::exception_ptr eptr) {
  try {
    if (eptr) std::rethrow_exception(eptr);
  } catch (const std::exception& e) {
    cout << "Error Caught (exception): " << e.what() << endl;
  } catch (...) {
    cout << "Error Caught (unknown exception)" << endl;
  }
}

void print_error(std::string err) {
  cout << "Error Caught (string): " << err << endl;
}

auto generic_error_handler = [](auto&& err) {
  cerr << "error enter" << endl;
  print_error(err);
};

int main() {
  using namespace std::chrono_literals;
  static static_thread_pool ctx{2};
  static timed_single_thread_context timed_ctx;
  inplace_stop_source stop_source;

  auto stop_token = stop_source.get_token();

  auto test_fn = [&stop_token]() {
    auto handle_sender = unifex::just_from([]() { cout << "Ready" << endl; });

    auto process_sender =
      unifex::stop_when(
        unifex::on(ctx.get_scheduler(), std::move(handle_sender)),
        unifex::on(
          timed_ctx.get_scheduler(),
          unifex::schedule_after(1s)
            | unifex::then([]() { cout << "Timeout triggered" << endl; })
            | unifex::upon_done([]() { cout << "Timeout cancelled" << endl; })))
      | unifex::then([]() { cout << "Result: done" << endl; });

    return process_sender;
  };

  auto test = with_query_value(test_fn(), get_stop_token, stop_token);

  unifex::v2::async_scope async_scope_;
  spawn_detached(
    with_query_value(std::move(test), get_stop_token, stop_token),
    async_scope_);
  sync_wait(async_scope_.join());

  return 0;
}