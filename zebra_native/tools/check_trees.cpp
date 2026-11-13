// Stage-1 parity check: score the exact model inputs captured from Python
// (export_model.py --reference) with the native evaluator and compare.
//
//   check_trees EXPORT_DIR
//
// For every base model and the FINAL model, reads reference/<NAME>.X.f64
// (rows x cols, float64), predicts, and compares with reference/<NAME>.p.f64.
// Also re-runs FINAL on base-model scores produced natively (end-to-end from
// the captured base-model inputs).
#include <cstdio>
#include <fstream>
#include <map>

#include "../src/gbdt.hpp"
#include "../third_party/nlohmann/json.hpp"

using json = nlohmann::json;

static std::vector<double> read_f64(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const auto n = static_cast<size_t>(in.tellg()) / sizeof(double);
  std::vector<double> v(n);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), n * sizeof(double));
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s EXPORT_DIR [reference|fuzz]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const std::string sub = argc == 3 ? argv[2] : "reference";
  const bool chain = sub == "reference";  // fuzz rows are independent per model
  json manifest = json::parse(std::ifstream(dir + "/manifest.json"));
  json ref = json::parse(std::ifstream(dir + "/" + sub + "/reference.json"));

  double worst = 0.0;
  std::map<std::string, std::vector<double>> native_scores;
  std::vector<json> models(manifest["base_models"].begin(), manifest["base_models"].end());
  models.push_back(manifest["final_model"]);

  zebra::Booster final_model;
  for (const auto& m : models) {
    const std::string name = m["name"];
    zebra::Booster b;
    b.load(dir + "/" + m["file"].get<std::string>());
    const auto& r = ref["calls"][name];
    const size_t rows = r["rows"], cols = r["cols"];
    if (cols != static_cast<size_t>(b.num_features) || cols != m["columns"].size()) {
      std::fprintf(stderr, "%s: column count mismatch (ref %zu, model %d, manifest %zu)\n",
                   name.c_str(), cols, b.num_features, m["columns"].size());
      return 1;
    }
    for (size_t j = 0; j < cols; ++j)
      if (r["columns"][j] != m["columns"][j]) {
        std::fprintf(stderr, "%s: column %zu name mismatch\n", name.c_str(), j);
        return 1;
      }
    const auto X = read_f64(dir + "/" + sub + "/" + name + ".X.f64");
    const auto P = read_f64(dir + "/" + sub + "/" + name + ".p.f64");
    double model_worst = 0.0;
    for (size_t i = 0; i < rows; ++i) {
      const double p = b.predict_proba(&X[i * cols]);
      native_scores[name].push_back(p);
      model_worst = std::max(model_worst, std::fabs(p - P[i]));
    }
    std::printf("%-15s trees=%5zu feats=%5zu sigmoid=%-18.15g max|native-python|=%.3g\n",
                name.c_str(), b.trees.size(), cols, b.sigmoid, model_worst);
    worst = std::max(worst, model_worst);
    if (name == "FINAL") final_model = std::move(b);
  }

  // Chain: rebuild FINAL inputs from native base scores (+ age, sex from captured FINAL X).
  if (chain) {
  const auto& fcols = manifest["final_model"]["columns"];
  const auto& fr = ref["calls"]["FINAL"];
  const size_t rows = fr["rows"], cols = fr["cols"];
  const auto FX = read_f64(dir + "/" + sub + "/FINAL.X.f64");
  const auto FP = read_f64(dir + "/" + sub + "/FINAL.p.f64");
  double chain_worst = 0.0;
  std::printf("\nend-to-end from base-model inputs (row order of FINAL call):\n");
  for (size_t i = 0; i < rows; ++i) {
    std::vector<double> x(FX.begin() + i * cols, FX.begin() + (i + 1) * cols);
    for (size_t j = 0; j < cols; ++j) {
      const std::string c = fcols[j];
      const auto suffix = c.rfind("_score");
      if (suffix != std::string::npos && suffix + 6 == c.size()) {
        // FINAL rows come from an inner join; match the base score by value of the
        // captured column (same patient), then substitute the native value.
        const auto& base = native_scores[c.substr(0, suffix)];
        const auto bp = read_f64(dir + "/" + sub + "/" + c.substr(0, suffix) + ".p.f64");
        size_t k = 0;
        while (k < bp.size() && bp[k] != x[j]) ++k;
        if (k == bp.size()) { std::fprintf(stderr, "cannot align %s row %zu\n", c.c_str(), i); return 1; }
        x[j] = base[k];
      }
    }
    const double p = final_model.predict_proba(x.data());
    chain_worst = std::max(chain_worst, std::fabs(p - FP[i]));
    std::printf("  row %zu  python=%.17g  native=%.17g\n", i, FP[i], p);
  }
  worst = std::max(worst, chain_worst);
  }
  std::printf("\nworst absolute difference: %.3g -> %s\n", worst, worst <= 1e-12 ? "PASS" : "FAIL");
  return worst <= 1e-12 ? 0 : 1;
}
