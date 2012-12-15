#include "common.h"

using namespace std;

// Graph
int V;                      // Number of vertices
vector<pair<int, int> > E;  // Edges
vector<vector<int> > A;     // Adjacent lists

// Layout
vector<vec3d> pos;
vector<double> color;
vector<double> vertex_alpha;

double norm(const vec3d &v) {
  double n = 0.0;
  for (size_t i = 0; i < v.size(); ++i) {
    n += v[i] * v[i];
  }
  return n;
}

double abs(const vec3d &v) {
  return sqrt(norm(v));
}

vec3d normalize(const vec3d &v) {
  return v / abs(v);
}

double dot(const vec3d &a, const vec3d &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
