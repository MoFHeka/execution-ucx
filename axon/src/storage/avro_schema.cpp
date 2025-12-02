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

#include <string_view>

#include "axon/storage/avro_schema.hpp"

namespace eux {
namespace axon {
namespace storage {

const std::string_view AvroSchema::kArrivalRequestSchema = R"({
  "type": "record",
  "name": "AxonRequest",
  "namespace": "eux.axon",
  "fields": [
    {
      "name": "header",
      "type": {
        "type": "record",
        "name": "RpcRequestHeader",
        "fields": [
          { "name": "function_id", "type": "int" },
          { "name": "session_id", "type": "int" },
          { "name": "request_id", "type": "int" },
          {
            "name": "params",
            "type": {
              "type": "array",
              "items": {
                "type": "record",
                "name": "ParamMeta",
                "fields": [
                  { "name": "type", "type": "int" },
                  {
                    "name": "value",
                    "type": [
                      "null", "boolean", "int", "long", "float", "double", "string",
                      {
                        "name": "BooleanArray",
                        "type": "record",
                        "fields": [
                          {
                            "name": "items",
                            "type": { "type": "array", "items": "boolean" }
                          }
                        ]
                      },
                      {
                        "name": "IntArray",
                        "type": "record",
                        "fields": [
                          {
                            "name": "items",
                            "type": { "type": "array", "items": "int" }
                          }
                        ]
                      },
                      {
                        "name": "LongArray",
                        "type": "record",
                        "fields": [
                          {
                            "name": "items",
                            "type": { "type": "array", "items": "long" }
                          }
                        ]
                      },
                      {
                        "name": "FloatArray",
                        "type": "record",
                        "fields": [
                          {
                            "name": "items",
                            "type": { "type": "array", "items": "float" }
                          }
                        ]
                      },
                      {
                        "name": "DoubleArray",
                        "type": "record",
                        "fields": [
                          {
                            "name": "items",
                            "type": { "type": "array", "items": "double" }
                          }
                        ]
                      },
                      {
                        "name": "TensorMeta",
                        "type": "record",
                        "fields": [
                          { "name": "device", "type": {
                              "name": "DLDevice", "type": "record",
                              "fields": [
                                { "name": "device_type", "type": "int" },
                                { "name": "device_id", "type": "int" }
                              ]
                            }
                          },
                          { "name": "ndim", "type": "int" },
                          { "name": "dtype", "type": {
                              "name": "DLDataType", "type": "record",
                              "fields": [
                                { "name": "code", "type": "int" },
                                { "name": "bits", "type": "int" },
                                { "name": "lanes", "type": "int" }
                              ]
                            }
                          },
                          { "name": "byte_offset", "type": "long" },
                          { "name": "shape", "type": { "type": "array", "items": "long" } },
                          { "name": "strides", "type": { "type": "array", "items": "long" } }
                        ]
                      }
                    ]
                  },
                  { "name": "name", "type": "string" }
                ]
              }
            }
          },
          {
            "name": "hlc",
            "type": {
              "name": "HybridLogicalClock",
              "type": "record",
              "fields": [{ "name": "raw", "type": "long" }]
            }
          },
          { "name": "workflow_id", "type": "int" }
        ]
      }
    },
    {
      "name": "payload",
      "type": [
        "null",
        {
          "type": "record",
          "name": "UcxBuffer",
          "fields": [
            { "name": "data", "type": "bytes" },
            { "name": "size", "type": "long" }
          ]
        },
        { "type": "array", "items": "UcxBuffer" }
      ]
    },
    {
      "name": "tensor_param_indices",
      "type": { "type": "array", "items": "long" }
    }
  ]
})";

const std::string_view AvroSchema::GetArrivalRequestSchema() {
  return kArrivalRequestSchema;
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
