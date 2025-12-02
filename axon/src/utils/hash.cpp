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

#include "axon/utils/hash.hpp"

#include <array>
#include <cstring>
#include <vector>

namespace eux {
namespace axon {
namespace utils {

namespace {

hash_t LoadHash(const uint8_t** data, const uint8_t* top) {
  std::ptrdiff_t size = top - (*data);
  if (size >= static_cast<int>(sizeof(hash_t))) {
    hash_t v;
    std::memcpy(&v, *data, sizeof(v));
    *data += sizeof(hash_t);
    return v;
  }
  using uval_t = union {
    hash_t h;
    std::array<uint8_t, sizeof(hash_t)> b{};
  };
#ifdef _MSC_VER
  // MSVC (or some versions we use) doesn't support C99 union field init
  // but it initializes the first member of the union.
  uval_t uval = {hash_t(0)};
#else
  uval_t uval = {.h = hash_t(0)};
#endif
  // use memcpy for compatibility with platforms not supporting unaligned access
  // note: compiled as single `movl` instr on x64.
  std::memcpy(uval.b.data(), *data, size);
  *data += size;
  return uval.h;
}

}  // namespace

hash_t HashBlock(const void* data, size_t n, const hash_t& seed) {
  const hash_t m(static_cast<uint64_t>(0xc6a4a7935bd1e995));
  const int r = 47;

  const uint8_t* u8_data = reinterpret_cast<const uint8_t*>(data);
  const uint8_t* top = u8_data + n;
  hash_t h(seed ^ (static_cast<uint64_t>(n * m)));
  while (u8_data < top) {
    hash_t k = LoadHash(&u8_data, top);
    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }
  h ^= h >> r;
  h *= m;
  h ^= h >> r;
  return h;
}

hash_t DataHash(const void* data, size_t size) {
  return HashBlock(
    data, size, hash_t(static_cast<uint64_t>(0xc2b2ae3d27d4eb4f)));
}

hash_t Hash(const std::vector<bool>& values) {
  // We can't assume a DataHash size/dataptr approach here bc
  // vector<bool> can be optimized as vector<bit> and storage details
  // are decoupled from actual size of 'bool' type
  hash_t h(static_cast<uint64_t>(0xad2ed1983bbf2e28));
  static const hash_t h_true(static_cast<uint64_t>(0x74f6b5198daa2b2));
  static const hash_t h_false(static_cast<uint64_t>(0xe39f30789cab5382));
  for (const auto& b : values) {
    if (b) {
      h = HashCombine(h, h_true);
    } else {
      h = HashCombine(h, h_false);
    }
  }
  return h;
}

}  // namespace utils
}  // namespace axon
}  // namespace eux
