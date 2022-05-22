/* Copyright 2021 Mikhail Isaev
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
#ifndef BRIDGE_HLO_PROF_COST_ANALYSIS_H_
#define BRIDGE_HLO_PROF_COST_ANALYSIS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "bridge/hlo_compute_cost_analysis.h"

class ProfCostAnalysis : public ComputeCostAnalysis {
 public:
  // Containers to store useful stats extracted from the profiler dump
  struct ProfiledInstructionStats {
    std::string name;
    int64_t occurrences;
    double time_us;
    double flops;
    double bytes_accessed;
  };

  ProfCostAnalysis(const xla::HloCostAnalysis::ShapeSizeFunction& shape_size,
                   bool instructions_from_trace,
                   bool time_from_trace,
                   bool loop_counter_from_trace,
                   const std::string& profiled_data_file,
                   int64_t num_cores,
                   const xla::HloCostAnalysis::Properties& per_second_rates);

  // Method that opens CSV file, reads its fields, and converts them to
  // ProfiledInstructionStats format writing into instruction_stats_map.
  xla::Status PopulateInstructionStatsFromProfiledData();

  // Update instructions properties using base class corresponding function, or
  // by using information from the trace, depending on
  xla::Status UpdateInstructionProperties();

 protected:
  enum class CSVState {
    UnquotedField,
    QuotedField,
    QuotedQuote
  };

  static constexpr size_t kCategoryIndex = 1;
  static constexpr size_t kHloNameIndex = 2;
  static constexpr size_t kOccurrencesIndex = 4;
  static constexpr size_t kTimeIndex = 8;
  static constexpr size_t kFlopsIndex = 12;
  static constexpr size_t kMemBWIndex = 13;

  // Map that keeps ProfiledInstructionStat entries for all the instructions
  // appeared in the trace or profiler dump
  absl::flat_hash_map<std::string, ProfiledInstructionStats>
    instruction_stats_map_;

  // Flags to choose between using statistics from ComputeCostAnalysis or from
  // profiled data
  bool instructions_from_trace_;
  bool time_from_trace_;
  bool loop_counters_from_trace_;

  // File name of CSV profiled data
  std::string filename_;

  int64_t num_cores_;

  // Method to read CSV line with quotes, from here
  // https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
  std::vector<std::string> ReadCSVRow(const std::string &row);

  // Method to parse CSV row and populate ProfiledInstructionStats for a single
  // instruction from the trace
  xla::Status ParseCSVRow(const std::vector<std::string>& row,
      ProfiledInstructionStats* instruction_stats);
};

#endif  // BRIDGE_HLO_PROF_COST_ANALYSIS_H_

