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
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "bridge/hlo_converter.h"
#include "paragraph/graph/graph.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

ABSL_FLAG(std::string, hlo_module, "",
          "A path to a file containing the HLO module.");
ABSL_FLAG(std::string, output_graph, "",
          "Output graph file.");
// defaults from V100 - 125 TF/s, 900 GB/s, NVLink 6 x 200 GB/s
// TPUv2 - 46 TF/s, 700 GB/s, ICI - 4 x 496 GB/s
// TPUv3 - 123 TF/s, 900 GB/s, ICI - 6 x 656 GB/s
// A100 - 312 TF/s, 1555/2039 GB/s, NVLink 600 GB/s
// data - https://cacm.acm.org/magazines/2020/7/245702-a-domain-specific-supercomputer-for-training-deep-neural-networks/fulltext
ABSL_FLAG(int64_t, num_replicas, 1,
          "The number of replicas the HLO module should be compiled for");
ABSL_FLAG(double, estimate_gibps, 900,
          "Estimate for GiB/s for processing unit's memory bandwidth.");
ABSL_FLAG(double, estimate_tflops, 125.0,
          "Estimate for TFLOPs/s for processing unit's performance.");
ABSL_FLAG(double, estimate_ttrps, 125.0,
          "Estimate for tera-transcendentals per second for processing unit");
ABSL_FLAG(bool, enforce_postorder, false,
    "Flag that enforces graph execution to be Post Order.");
ABSL_FLAG(bool, profiled_data, false,
    "Flag that enables populating graph with statistics from profiled run");
ABSL_FLAG(bool, instructions_from_trace, true,
    "Flag that sets to 0 instructions time if they don't appear in the trace, "
    "works only if 'profiled_data' flag set to 'true'.");
ABSL_FLAG(bool, loop_counters_from_trace, true,
    "Flag that enforces populating loop counters from profiled data, "
    "works only if 'profiled_data' flag set to 'true'.");
ABSL_FLAG(bool, time_from_trace, true,
    "Flag that enforces populating instruction timings from profiled data, "
    "works only if 'profiled_data' flag set to 'true'.");
ABSL_FLAG(std::string, profiled_data_file, "",
          "File path to the profiled data dump.");

int32_t main(int32_t argc, char** argv) {
  // Parsing flags
  absl::ParseCommandLine(argc, argv);

  std::string module_path = absl::GetFlag(FLAGS_hlo_module);
  CHECK_NE(module_path, "");
  std::string output_graph_file = absl::GetFlag(FLAGS_output_graph);
  CHECK_NE(output_graph_file, "");
  std::string target_extension = std::filesystem::path(
      output_graph_file).extension();
  CHECK((target_extension == paragraph::Graph::kBinaryProtoExtension ||
         target_extension == paragraph::Graph::kTextProtoExtension));
  int64_t num_replicas = absl::GetFlag(FLAGS_num_replicas);

  double flops = absl::GetFlag(FLAGS_estimate_tflops) * 1e12;
  double trps = absl::GetFlag(FLAGS_estimate_ttrps) * 1e12;
  double bps = absl::GetFlag(FLAGS_estimate_gibps) * 1024 * 1024 * 1024;
  const xla::HloCostAnalysis::Properties perf_prop = {
    { xla::HloCostAnalysis::kFlopsKey, flops },
    { xla::HloCostAnalysis::kTranscendentalsKey, trps },
    { xla::HloCostAnalysis::kBytesAccessedKey, bps }
  };
  bool enforce_postorder = absl::GetFlag(FLAGS_enforce_postorder);

  std::string profiled_data_file = absl::GetFlag(FLAGS_profiled_data_file);
  bool profiled_data = absl::GetFlag(FLAGS_profiled_data);
  bool instructions_from_trace = absl::GetFlag(FLAGS_instructions_from_trace);
  bool loop_counters_from_trace = absl::GetFlag(FLAGS_loop_counters_from_trace);
  bool time_from_trace = absl::GetFlag(FLAGS_time_from_trace);

  std::unique_ptr<xla::HloModule> module;
  xla::hlo_module_loader_details::Config loader_config;
  loader_config.num_replicas = num_replicas;
  auto module_statusor = xla::LoadModuleFromFile(module_path, loader_config);
  TF_CHECK_OK(module_statusor.status());
  module = std::move(module_statusor.ValueOrDie());

  auto graph_proto_statusor = HloConverter(
      module.get(), num_replicas, perf_prop,
      profiled_data, profiled_data_file, instructions_from_trace,
      time_from_trace, loop_counters_from_trace);
  TF_CHECK_OK(graph_proto_statusor.status());
  auto graph_proto = graph_proto_statusor.ValueOrDie();
  auto graph_statusor = paragraph::Graph::CreateFromProto(graph_proto,
                                                          /*reset_ids =*/ true);
  CHECK_OK(graph_statusor.status());
  std::unique_ptr<paragraph::Graph> graph = std::move(graph_statusor.value());
  if (enforce_postorder) {
    graph->PostOrderEnforcer();
  }

  CHECK_OK(graph->WriteToFile(output_graph_file));
  return 0;
}
