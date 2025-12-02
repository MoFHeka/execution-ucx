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

#ifndef AXON_CORE_STORAGE_AVRO_SCHEMA_HPP_
#define AXON_CORE_STORAGE_AVRO_SCHEMA_HPP_

#include <string_view>

namespace eux {
namespace axon {
namespace storage {

/**
 * @brief Avro schema definition for AxonRequest serialization
 *
 * This schema defines the structure for serializing AxonRequest objects
 * to Avro format for persistent storage and sharing.
 */
class AvroSchema {
 public:
  /**
   * @brief Get the Avro schema JSON for AxonRequest
   * @return JSON string containing the Avro schema definition
   */
  static const std::string_view GetArrivalRequestSchema();

 private:
  static const std::string_view kArrivalRequestSchema;
};

}  // namespace storage
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_STORAGE_AVRO_SCHEMA_HPP_
