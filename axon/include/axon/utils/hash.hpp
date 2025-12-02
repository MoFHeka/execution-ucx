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

#ifndef AXON_CORE_UTILS_HASH_HPP_
#define AXON_CORE_UTILS_HASH_HPP_

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace eux {
namespace axon {
namespace utils {

// uint128 wrapper for hash_t, supports stream output and native-like operations
struct hash_t {
  unsigned __int128 value;

  // Constructors
  hash_t() : value(0) {}
  explicit hash_t(unsigned __int128 v) : value(v) {}
  // Implicit conversion from 64-bit integer types
  explicit hash_t(uint64_t v) : value(static_cast<unsigned __int128>(v)) {}
  explicit hash_t(uint32_t v)
    : value(static_cast<unsigned __int128>(static_cast<uint64_t>(v))) {}
  explicit hash_t(int v)
    : value(static_cast<unsigned __int128>(static_cast<uint64_t>(v))) {}
  hash_t(uint64_t high, uint64_t low)
    : value((static_cast<unsigned __int128>(high) << 64) | low) {}

  // Static factory methods
  static hash_t from_u64(uint64_t v) { return hash_t(v); }
  static hash_t from_u32(uint32_t v) { return hash_t(v); }
  static hash_t from_i32(int v) { return hash_t(v); }
  static hash_t from_u128(unsigned __int128 v) { return hash_t(v); }
  static hash_t from_high_low(uint64_t high, uint64_t low) {
    return hash_t(high, low);
  }

  uint64_t high() const { return static_cast<uint64_t>(value >> 64); }
  uint64_t low() const {
    return static_cast<uint64_t>(value & 0xFFFFFFFFFFFFFFFFULL);
  }

  // Conversion
  operator unsigned __int128() const { return value; }

  // Comparison operators
  bool operator==(const hash_t& other) const { return value == other.value; }
  bool operator!=(const hash_t& other) const { return value != other.value; }
  bool operator<(const hash_t& other) const { return value < other.value; }
  bool operator>(const hash_t& other) const { return value > other.value; }
  bool operator<=(const hash_t& other) const { return value <= other.value; }
  bool operator>=(const hash_t& other) const { return value >= other.value; }

  // Arithmetic operators
  hash_t operator+(const hash_t& rhs) const {
    return hash_t(value + rhs.value);
  }
  hash_t operator-(const hash_t& rhs) const {
    return hash_t(value - rhs.value);
  }
  hash_t operator*(const hash_t& rhs) const {
    return hash_t(value * rhs.value);
  }
  hash_t operator/(const hash_t& rhs) const {
    return hash_t(value / rhs.value);
  }
  hash_t operator%(const hash_t& rhs) const {
    return hash_t(value % rhs.value);
  }

  // Bitwise operators
  hash_t operator~() const { return hash_t(~value); }
  hash_t operator&(const hash_t& rhs) const {
    return hash_t(value & rhs.value);
  }
  hash_t operator|(const hash_t& rhs) const {
    return hash_t(value | rhs.value);
  }
  hash_t operator^(const hash_t& rhs) const {
    return hash_t(value ^ rhs.value);
  }
  hash_t operator<<(int n) const { return hash_t(value << n); }
  hash_t operator>>(int n) const { return hash_t(value >> n); }

  // Compound assignment operators
  hash_t& operator+=(const hash_t& rhs) {
    value += rhs.value;
    return *this;
  }
  hash_t& operator-=(const hash_t& rhs) {
    value -= rhs.value;
    return *this;
  }
  hash_t& operator*=(const hash_t& rhs) {
    value *= rhs.value;
    return *this;
  }
  hash_t& operator/=(const hash_t& rhs) {
    value /= rhs.value;
    return *this;
  }
  hash_t& operator%=(const hash_t& rhs) {
    value %= rhs.value;
    return *this;
  }
  hash_t& operator&=(const hash_t& rhs) {
    value &= rhs.value;
    return *this;
  }
  hash_t& operator|=(const hash_t& rhs) {
    value |= rhs.value;
    return *this;
  }
  hash_t& operator^=(const hash_t& rhs) {
    value ^= rhs.value;
    return *this;
  }
  hash_t& operator<<=(int n) {
    value <<= n;
    return *this;
  }
  hash_t& operator>>=(int n) {
    value >>= n;
    return *this;
  }

  // Unary operators
  hash_t operator+() const { return *this; }
  hash_t operator-() const { return hash_t(-value); }
};

// Stream output operator for hash_t, outputs as hex string
inline std::ostream& operator<<(std::ostream& os, const hash_t& h) {
  uint64_t high = static_cast<uint64_t>(h >> 64);
  uint64_t low = static_cast<uint64_t>(h & 0xFFFFFFFFFFFFFFFFULL);
  os << std::hex << high << std::setfill('0') << low;
  return os;
}

// HashCombine for hash_t
inline uint64_t Hash64(uintmax_t low64, uintmax_t high64) {
  // Combine the high and low 64-bit parts of the hash.
  // This is a variant of boost::hash_combine.
  static constexpr uint64_t kHash64Mul = 0x27d4eb2f165667c5;
  // A large prime derived from the golden ratio.
  static constexpr uint64_t kHash64GoldenRatio = 0x9e3779b97f4a7c15;

  uint64_t seed = high64 * kHash64Mul;
  seed += kHash64GoldenRatio;
  seed += (low64 << 6) + (low64 >> 2);
  return low64 ^ seed;
}

inline hash_t Hash128(const hash_t& a, const hash_t& b) {
  // Combine the two 128-bit hash.
  static const hash_t kHash128Mul(101, 0x27d4eb2f165667c5);
  // A large prime derived from the golden ratio.
  static const hash_t kHash128GoldenRatio(
    static_cast<uint64_t>(0x9e3779b97f4a7c15));

  hash_t seed = b * kHash128Mul;
  seed += kHash128GoldenRatio;
  seed += (a << 6) + (a >> 2);
  return a ^ seed;
}

inline hash_t HashCombine(const hash_t& a, const hash_t& b) {
  return Hash128(a, b);
}

inline hash_t HashCombine(const hash_t& a, uintmax_t b) {
  return Hash128(a, hash_t::from_u64(b));
}

inline hash_t HashCombine(uintmax_t a, const hash_t& b) {
  return Hash128(hash_t::from_u64(a), b);
}

inline hash_t HashCombine(uintmax_t a, uintmax_t b) {
  return Hash128(hash_t::from_u64(a), hash_t::from_u64(b));
}

hash_t DataHash(const void* data, size_t size);

static inline hash_t StringHash(const std::string& str) {
  return DataHash(str.data(), str.size());
}

static inline hash_t StringHash(std::string_view str) {
  return DataHash(str.data(), str.size());
}

static inline hash_t StringHash(const char* str) {
  return DataHash(str, std::strlen(str));
}

// Automatic templated implementation for 'arithmetic' types
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
hash_t Hash(const T& value) {
  return DataHash(&value, sizeof(value));
}

hash_t Hash(const std::vector<bool>& values);

static inline hash_t Hash(const std::string& value) {
  return DataHash(value.data(), value.size());
}

static inline hash_t Hash(std::string_view value) {
  return DataHash(value.data(), value.size());
}

// Taken from glibc's implementation of hashing optionals,
// we want to include a contribution to the hash to distinguish
// cases where one or another option was null, but we hope it doesn't
// collide with an actually scalar value.
//
// Use an arbitrary randomly-selected 64-bit integer rather than a
// small constant that we then hash at runtime so we don't have to
// repeatedly hash a constant at runtime.
static const uint64_t kNullOpt = 0x8655d738f3678dda;

// Need a special case for std::optional<container>?
template <typename T>
hash_t Hash(const std::optional<std::vector<T>>& value) {
  if (value.has_value()) {
    return ContainerHash(value.value());
  } else {
    return hash_t(kNullOpt);
  }
}

// Hashing of containers
// Forward declare to allow hashes of vectors of vectors to work.
template <typename T>
hash_t ContainerHash(const T& values);

template <typename T>
hash_t Hash(const std::vector<T>& values) {
  return ContainerHash(values);
}

template <typename T>
hash_t Hash(const std::set<T>& values) {
  return ContainerHash(values);
}

template <typename T, typename S>
hash_t Hash(const std::pair<T, S>& values) {
  return HashCombine(Hash(values.first), Hash(values.second));
}

static inline hash_t Hash(const hash_t& value) { return value; }

template <typename T>
hash_t ContainerHash(const T& values) {
  hash_t h(static_cast<uint64_t>(0x85ebca77c2b2ae63));
  for (const auto& value : values) {
    h = HashCombine(h, Hash(value));
  }
  return h;
}

// Varargs hashing
template <typename T = void>
hash_t MHash() {
  return hash_t(static_cast<uint64_t>(0x165667b19e3779f9));
}

// When specializing Hash(T) also specialize MHash(T, ...) since
template <typename T, typename... Targs>
hash_t MHash(const T& value, Targs... Fargs) {
  return HashCombine(Hash(value), MHash(Fargs...));
}

}  // namespace utils
}  // namespace axon
}  // namespace eux

namespace std {
template <>
struct hash<eux::axon::utils::hash_t> {
  size_t operator()(const eux::axon::utils::hash_t& k) const {
    return eux::axon::utils::Hash64(k.low(), k.high());
  }
};
}  // namespace std

#endif  // AXON_CORE_UTILS_HASH_HPP_
