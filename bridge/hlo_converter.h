/* Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BRIDGE_HLO_CONVERTER_H_
#define BRIDGE_HLO_CONVERTER_H_

#include <string>

#include "bridge/hlo_compute_cost_analysis.h"
#include "paragraph/graph/graph.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

xla::StatusOr<paragraph::GraphProto> HloConverter(
      const xla::HloModule* module, int64_t num_cores,
      const ComputeCostAnalysis::Properties& per_second_rates,
      bool profiled_data = false, const std::string profiled_data_file = "",
      bool instructions_from_trace = true,
      bool time_from_trace = true, bool loop_counters_from_trace = true);

#endif  // BRIDGE_HLO_CONVERTER_H_
