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
#include "bridge/hlo_prof_cost_analysis.h"

#include <math.h>

#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"

ProfCostAnalysis::ProfCostAnalysis(
    const ShapeSizeFunction& shape_size,
    bool time_from_trace,
    bool loop_counters_from_trace,
    const std::string& profiled_data_file,
    const Properties& per_second_rates)
  : ComputeCostAnalysis(shape_size, per_second_rates) {
    filename_ = profiled_data_file;
    time_from_trace_ = time_from_trace;
    loop_counters_from_trace_ = loop_counters_from_trace;
  }

xla::Status ProfCostAnalysis::PopulateInstructionStatsFromProfiledData() {
  TF_RET_CHECK(filename_ != "");
  std::ifstream csv_stream;
  csv_stream.open(filename_, std::ios::in);
  TF_RET_CHECK(csv_stream.is_open());

  std::string row;
  while (!csv_stream.eof()) {
    std::getline(csv_stream, row);
    TF_RET_CHECK(!csv_stream.bad() && !csv_stream.fail());
    auto fields = ReadCSVRow(row);
    ProfCostAnalysis::ProfiledInstructionStats instruction_stats;
    TF_CHECK_OK(ParseCSVRow(fields, &instruction_stats));
    TF_RET_CHECK(instruction_stats_map_.emplace(
          instruction_stats.name, instruction_stats).second);
  }
  return xla::Status::OK();
}

xla::Status ProfCostAnalysis::UpdateInstructionProperties() {
  // First read the data from profiled run
  TF_CHECK_OK(PopulateInstructionStatsFromProfiledData());
  // Second, use HloCostAnalysis if corresponding flag is passed
  if (!time_from_trace_) {
    TF_CHECK_OK(ComputeCostAnalysis::UpdateInstructionProperties());
  }
  for (auto& map_it : hlo_properties_) {
    const xla::HloInstruction* hlo = map_it.first;
    const std::string hlo_name = hlo->name();
    // If metrics are populated from trace, override CostAnalysis defaults
    if (time_from_trace_) {
      Properties hlo_property;
      if (instruction_stats_map_.find(hlo_name) !=
          instruction_stats_map_.end()) {
        hlo_property = {
          {kFlopsKey, instruction_stats_map_[hlo_name].flops},
          {kTranscendentalsKey, 0L},
          {kBytesAccessedKey, instruction_stats_map_[hlo_name].bytes_accessed},
          {kOperandBytesAccessedKey, 0L},
          {kOutputBytesAccessedKey, 0L},
          {kOptimalSecondsKey,
            instruction_stats_map_[hlo_name].time_us * 0.000001}};
      } else {
        hlo_property = {
          {kFlopsKey, 0L},
          {kTranscendentalsKey, 0L},
          {kBytesAccessedKey, 0L},
          {kOperandBytesAccessedKey, 0L},
          {kOutputBytesAccessedKey, 0L},
          {kOptimalSecondsKey, 0L}};
      }
      TF_RET_CHECK(instruction_properties_.emplace(hlo, hlo_property).second);
    }
    TF_RET_CHECK(instruction_properties_.find(hlo) !=
                 instruction_properties_.end());
    Properties& hlo_property = instruction_properties_.at(hlo);
    // Zero instruction time if it has 0 occurrences
    if (instruction_stats_map_[hlo_name].occurrences == 0) {
      hlo_property[kOptimalSecondsKey] = 0;
    }
    // Populate loop counters
    if (hlo->opcode() == xla::HloOpcode::kWhile) {
      if (loop_counters_from_trace_) {
        uint64_t children_count = 0;
        uint64_t children_occurrences = 0;
        for (const auto& child :
            hlo->called_computations().at(0)->instructions()) {
          if (instruction_stats_map_.find(child->name()) !=
              instruction_stats_map_.end()) {
            children_count += 1;
            children_occurrences +=
              instruction_stats_map_[child->name()].occurrences;
          }
        }
        if (children_count > 0) {
          hlo_property[kOccurrencesKey] = round(
              children_occurrences / children_count);
        } else {
          hlo_property[kOccurrencesKey] = 1;
        }
      }
    }
  }
  return xla::Status::OK();
}

std::vector<std::string> ProfCostAnalysis::ReadCSVRow(const std::string &row) {
  CSVState state = CSVState::UnquotedField;
  std::vector<std::string> fields {""};
  size_t i = 0;
  for (char c : row) {
    switch (state) {
      case CSVState::UnquotedField:
        switch (c) {
          case ',':
            fields.push_back(""); i++;
            break;
          case '"':
            state = CSVState::QuotedField;
            break;
          default:
            fields[i].push_back(c);
            break;
        }
        break;
      case CSVState::QuotedField:
        switch (c) {
          case '"':
            state = CSVState::QuotedQuote;
            break;
          default:
            fields[i].push_back(c);
            break;
        }
        break;
      case CSVState::QuotedQuote:
        switch (c) {
          case ',':
            fields.push_back(""); i++;
            state = CSVState::UnquotedField;
            break;
          case '"':
            fields[i].push_back('"');
            state = CSVState::QuotedField;
            break;
          default:
            state = CSVState::UnquotedField;
            break;
        }
        break;
    }
  }
  return fields;
}

xla::Status ProfCostAnalysis::ParseCSVRow(
    const std::vector<std::string>& row,
    ProfCostAnalysis::ProfiledInstructionStats* instruction_stats) {
  ProfiledInstructionStats stats;
  std::string hlo_string = row.at(kHloNameIndex);
  std::vector<std::string> hlo_split = absl::StrSplit(hlo_string, ' ');
  stats.name = hlo_split.at(0).substr(1);

  std::string occurrences_string = row.at(kOccurrencesIndex);
  TF_RET_CHECK(absl::SimpleAtoi(occurrences_string,
      &(instruction_stats->occurrences)));

  std::string time_string = row.at(kTimeIndex);
  TF_RET_CHECK(absl::SimpleAtod(time_string,
        &(instruction_stats->time_us)));

  std::string gflops_sec_string = row.at(kFlopsIndex);
  double gflops_from_string;
  TF_RET_CHECK(absl::SimpleAtod(gflops_sec_string, &gflops_from_string));
  instruction_stats->flops = gflops_from_string * stats.time_us / 1000;

  std::string gbytes_sec_string = row.at(kMemBWIndex);
  double gbytes_from_string;
  TF_RET_CHECK(absl::SimpleAtod(gbytes_sec_string, &gbytes_from_string));
  instruction_stats->bytes_accessed = gbytes_from_string * stats.time_us / 1000;
  return xla::Status::OK();
}
