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

#ifndef AXON_CORE_STORAGE_AVRO_SERIALIZATION_HPP_
#define AXON_CORE_STORAGE_AVRO_SERIALIZATION_HPP_

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <avro/Generic.hh>

#include "axon/storage/avro_schema.hpp"
#include "axon/utils/axon_message.hpp"

namespace eux {
namespace axon {
namespace storage {

using eux::ucxx::UcxMemoryResourceManager;

using AxonRequest = eux::axon::utils::AxonRequest;

/**
 * @brief Avro serialization utilities for AxonRequest
 *
 * This class provides methods to serialize and deserialize AxonRequest
 * objects to/from Avro binary format for persistent storage.
 */
class AvroSerialization {
 public:
  /**
   * @brief Serialize an AxonRequest to Avro binary format
   * @param request The AxonRequest to serialize
   * @return Vector containing the serialized binary data
   */
  static avro::GenericDatum Serialize(
    const AxonRequest& request, UcxMemoryResourceManager& mr);

  /**
   * @brief Deserialize an AxonRequest from Avro binary format
   * @param data The binary data to deserialize
   * @return Deserialized AxonRequest
   */
  static std::shared_ptr<AxonRequest> Deserialize(
    const avro::GenericDatum& data, UcxMemoryResourceManager& mr);

  /**
   * @brief Serialize an AxonRequest to Avro JSON format
   * @param request The AxonRequest to serialize
   * @return JSON string representation
   */
  static std::string SerializeToJson(const AxonRequest& request);

  /**
   * @brief Deserialize an AxonRequest from Avro JSON format
   * @param json_data The JSON string to deserialize
   * @return Deserialized AxonRequest
   */
  static AxonRequest DeserializeFromJson(const std::string& json_data);
};

}  // namespace storage
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_STORAGE_AVRO_SERIALIZATION_HPP_
