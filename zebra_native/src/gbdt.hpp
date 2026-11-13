// Minimal LightGBM (v3.x text format) binary-classifier evaluator.
//
// Reproduces LightGBM's Tree::NumericalDecision and GBDT::Predict for
// objective=binary with numerical splits. Categorical splits and linear trees
// are rejected at load time rather than evaluated incorrectly.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zebra {

struct Tree {
  int num_leaves = 1;
  std::vector<int> split_feature;
  std::vector<double> threshold;
  std::vector<int8_t> decision_type;
  std::vector<int> left_child, right_child;
  std::vector<double> leaf_value;

  // LightGBM: tree.h, NumericalDecision()
  int next(int node, double fval) const {
    const int8_t dt = decision_type[node];
    const int missing_type = (dt >> 2) & 3;  // 0 None, 1 Zero, 2 NaN
    const bool default_left = (dt & 2) != 0;
    if (std::isnan(fval) && missing_type != 2) fval = 0.0;
    if ((missing_type == 1 && fval > -1e-35 && fval < 1e-35) ||
        (missing_type == 2 && std::isnan(fval))) {
      return default_left ? left_child[node] : right_child[node];
    }
    return fval <= threshold[node] ? left_child[node] : right_child[node];
  }

  double predict(const double* x) const {
    if (num_leaves <= 1) return leaf_value[0];
    int node = 0;
    while (node >= 0) node = next(node, x[split_feature[node]]);
    return leaf_value[~node];
  }
};

class Booster {
 public:
  std::vector<Tree> trees;
  int num_features = 0;  // max_feature_idx + 1
  double sigmoid = 1.0;

  // The sigmoid is taken from the file's `objective=binary sigmoid:X` line. It is
  // printed with ~6 significant digits, but a pickled LightGBM model is rebuilt
  // from the same text, so this is the value the Python model really uses.
  void load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    Tree* cur = nullptr;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line == "end of trees") break;
      if (line.rfind("Tree=", 0) == 0) {
        trees.emplace_back();
        cur = &trees.back();
        continue;
      }
      const auto eq = line.find('=');
      if (eq == std::string::npos) continue;
      const std::string key = line.substr(0, eq), val = line.substr(eq + 1);
      if (!cur) {
        if (key == "max_feature_idx") num_features = std::stoi(val) + 1;
        else if (key == "objective") parse_objective(val);
        else if (key == "num_class" && std::stoi(val) != 1)
          throw std::runtime_error(path + ": multiclass models not supported");
        continue;
      }
      if (key == "num_leaves") cur->num_leaves = std::stoi(val);
      else if (key == "num_cat" && std::stoi(val) != 0)
        throw std::runtime_error(path + ": categorical splits not supported");
      else if (key == "is_linear" && std::stoi(val) != 0)
        throw std::runtime_error(path + ": linear trees not supported");
      else if (key == "split_feature") cur->split_feature = ints<int>(val);
      else if (key == "threshold") cur->threshold = doubles(val);
      else if (key == "decision_type") cur->decision_type = ints<int8_t>(val);
      else if (key == "left_child") cur->left_child = ints<int>(val);
      else if (key == "right_child") cur->right_child = ints<int>(val);
      else if (key == "leaf_value") cur->leaf_value = doubles(val);
    }
    for (const auto& t : trees) {
      for (auto d : t.decision_type)
        if (d & 1) throw std::runtime_error(path + ": categorical split found");
      if (t.leaf_value.size() != static_cast<size_t>(t.num_leaves))
        throw std::runtime_error(path + ": malformed tree");
    }
    if (trees.empty()) throw std::runtime_error(path + ": no trees");
  }

  double raw(const double* x) const {
    double s = 0.0;
    for (const auto& t : trees) s += t.predict(x);  // same order as LightGBM
    return s;
  }

  // LightGBM BinaryLogloss::ConvertOutput
  double predict_proba(const double* x) const {
    return 1.0 / (1.0 + std::exp(-sigmoid * raw(x)));
  }

 private:
  void parse_objective(const std::string& v) {
    std::istringstream ss(v);
    std::string tok;
    ss >> tok;
    if (tok != "binary") throw std::runtime_error("objective '" + tok + "' not supported");
    while (ss >> tok)
      if (tok.rfind("sigmoid:", 0) == 0) sigmoid = std::strtod(tok.c_str() + 8, nullptr);
  }
  template <class T>
  static std::vector<T> ints(const std::string& v) {
    std::vector<T> out;
    std::istringstream ss(v);
    long x;
    while (ss >> x) out.push_back(static_cast<T>(x));
    return out;
  }
  static std::vector<double> doubles(const std::string& v) {
    std::vector<double> out;
    const char* p = v.c_str();
    char* end;
    for (;;) {
      double d = std::strtod(p, &end);
      if (end == p) break;
      out.push_back(d);
      p = end;
    }
    return out;
  }
};

}  // namespace zebra
