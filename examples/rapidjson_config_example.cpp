#include <cstdint>
#include <iostream>
#include <string>

#include "sim/rng.hpp"

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

namespace {

struct SimConfig {
    std::uint64_t seed = 42;
    std::uint32_t events = 1000;
    std::string scenario = "baseline";
};

SimConfig parse_config_json(const std::string& json) {
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError() || !doc.IsObject()) {
        return {};
    }

    SimConfig cfg;
    if (doc.HasMember("seed") && doc["seed"].IsUint64()) {
        cfg.seed = doc["seed"].GetUint64();
    }
    if (doc.HasMember("events") && doc["events"].IsUint()) {
        cfg.events = doc["events"].GetUint();
    }
    if (doc.HasMember("scenario") && doc["scenario"].IsString()) {
        cfg.scenario = doc["scenario"].GetString();
    }
    return cfg;
}

std::string make_result_json(const SimConfig& cfg, std::uint64_t checksum) {
    rapidjson::Document out;
    out.SetObject();
    auto& alloc = out.GetAllocator();

    out.AddMember("seed", rapidjson::Value().SetUint64(cfg.seed), alloc);
    out.AddMember("events", rapidjson::Value().SetUint(cfg.events), alloc);
    out.AddMember("scenario", rapidjson::Value(cfg.scenario.c_str(), alloc), alloc);
    out.AddMember("rng_checksum", rapidjson::Value().SetUint64(checksum), alloc);

    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    out.Accept(writer);
    return buffer.GetString();
}

} // namespace

int main() {
    const std::string config_json = R"json(
{
  "seed": 1337,
  "events": 16,
  "scenario": "toy"
}
)json";

    const SimConfig cfg = parse_config_json(config_json);

    sim::Rng rng(cfg.seed);
    std::uint64_t checksum = 0;
    for (std::uint32_t i = 0; i < cfg.events; ++i) {
        checksum ^= rng.next_u64();
    }

    const std::string result_json = make_result_json(cfg, checksum);
    std::cout << result_json << '\n';
    return 0;
}
