#include "common.h"
#include <malloc.h>
#include <emmintrin.h>
#include <smmintrin.h>
// #include "redsvd/redsvd.hpp"

using namespace std;

namespace {
// (x,y,z) / sqrt(x^2 + y^2 + z^2)
inline __m128 normalize3(__m128 v)
{
  __m128 inverse_norm = _mm_rsqrt_ps(_mm_dp_ps(v, v, 0x77));
  return _mm_mul_ps(v, inverse_norm);
}
}  // namespace

namespace layout {
vector<vec3d> speed;

namespace naive_force {
const double D = 0.999;
const double K = 1.0;
const double B = 0.5;
const double G = 0.05;
const double F = 0.5;
const double R = 5;
double d = 0.1;

double Step() {
#pragma omp flush

  benchmark("iteration") {
#pragma omp parallel for
  for (int i = 0; i < V; ++i) {
    vec3d force(0.0, 3);

    sort(A[i].begin(), A[i].end());

    for (int k = 0; k < V; ++k) {
      int j = k;
      if (i == j) continue;
      vec3d d = pos[j] - pos[i];
      double a = abs(d);
      force -= d * (B / E.size() / (EPS + a * a * a) * A[j].size());
    }

    for (size_t k = 0; k < A[i].size(); ++k) {
      int j = A[i][k];
      if (i == j) continue;
      vec3d d = pos[j] - pos[i];
      force += d * (K / A[i].size());
    }

    force -= pos[i] * G;
    speed[i] += force;
    speed[i] *= 1 - F;

    {
      double a = abs(speed[i]);
      if (a > 0.4) speed[i] *= 0.4 / a;
      // if (a > d) speed[i] *= d / a;
    }

    //double a = abs(speed[i]);
    // if (a > d) speed[i] *= d / a;
    // if (a > 10) speed[i] *= 10 / a;
  }

#pragma omp flush

  for (int i = 0; i < V; ++i) {
    pos[i] += 0.4 * speed[i];
    if (abs(pos[i]) > R) pos[i] *= R / abs(pos[i]);
  }
  }

  d *= D;
  return 0.0;
}


double Step2() {
#pragma omp flush

#pragma omp parallel for
  for (int i = 0; i < V; ++i) {
    vec3d force(0.0, 3);

    sort(A[i].begin(), A[i].end());

    for (int k = 0; k < V; ++k) {
      int j = k;
      if (i == j) continue;
      vec3d d = pos[j] - pos[i];
      double a = abs(d);
      force -= d * B / (EPS + a * a * a);
    }

    for (size_t k = 0; k < A[i].size(); ++k) {
      int j = A[i][k];
      if (i == j) continue;
      vec3d d = pos[j] - pos[i];
      force += d * K;
    }

    force -= pos[i] * G;
    speed[i] += force;
    speed[i] *= 1 - F;

    //double a = abs(speed[i]);
    // if (a > d) speed[i] *= d / a;
    // if (a > 10) speed[i] *= 10 / a;
  }

#pragma omp flush

  for (int i = 0; i < V; ++i) {
    pos[i] += 0.01 * speed[i];
    if (abs(pos[i]) > R) pos[i] *= R / abs(pos[i]);
  }

  d *= D;
  return 0.0;
}


double Step3() {
#pragma omp flush

  benchmark("iteration") {
    vec3d g(0.0, 3);
    for (int i = 0; i < V; ++i) g += pos[i];
    g /= V;

#pragma omp parallel for
  for (int i = 0; i < V; ++i) {
    vec3d force(0.0, 3);

    // for (int k = 0; k < V; ++k) {
    //   int j = k;
    //   if (i == j) continue;
    //   vec3d d = pos[j] - pos[i];
    //   double a = abs(d);
    //   force -= d * (B / E.size() / (EPS + a * a * a) * A[j].size());
    // }

    vec3d d = pos[i] - g;
    double a = abs(d);
    force += d * B / (EPS + a * a * a) * 0.3;

    for (size_t k = 0; k < A[i].size(); ++k) {
      int j = A[i][k];
      if (i == j) continue;
      vec3d d = pos[j] - pos[i];
      force += d * (K / A[i].size());
    }

    force -= pos[i] * G;
    speed[i] += force;
    speed[i] *= 1 - F;

  }

#pragma omp flush

  for (int i = 0; i < V; ++i) {
    pos[i] += 0.4 * speed[i];
    // if (abs(pos[i]) > R) pos[i] *= R / abs(pos[i]);
  }
  }

  d *= D;
  return 0.0;
}
}  // namespace naive_force

namespace simd_naive_force {
const double D = 0.999;
const double K = 1.0;
const double B = 0.5;
const double G = 0.05;
const double F = 0.5;
const double R = 5;
double d = 0.1;

double SIMDStep() {
  float *pos_m128 = (float*)memalign(64, sizeof(__m128) * V * 4);
  for (int i = 0; i < V; ++i) {
    for (int j = 0; j < 3; ++j) pos_m128[i * 4 + j] = pos[i][j];
    pos_m128[i * 4 + 3] = 0;
  }

  #pragma omp flush

  benchmark("SIMD iteration") {
    #pragma omp parallel for
  for (int i = 0; i < V; ++i) {
    __m128 force = _mm_setzero_ps();
    __m128 my_pos = _mm_load_ps(pos_m128 + i * 4);

    for (int k = 0; k < V; ++k) {
      int j = k;
      if (i == j) continue;
      __m128 op_pos = _mm_load_ps(pos_m128 + j * 4);
      __m128 d = _mm_sub_ps(my_pos, op_pos);
      float a = _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(d, d, 0x71)));

      float t = B / E.size() / (EPS + a * a * a) * A[j].size();
      force = _mm_add_ps(force, _mm_mul_ps(d, _mm_set1_ps(t)));
    }

    for (size_t k = 0; k < A[i].size(); ++k) {
      int j = A[i][k];
      if (i == j) continue;
      __m128 op_pos = _mm_load_ps(pos_m128 + j * 4);
      __m128 d = _mm_sub_ps(op_pos, my_pos);
      float t = K / A[i].size();
      force = _mm_add_ps(force, _mm_mul_ps(d, _mm_set1_ps(t)));
    }

    {
      force = _mm_sub_ps(force, _mm_mul_ps(my_pos, _mm_set1_ps((float)G)));
    }

    float tmp[4];
    _mm_storeu_ps(tmp, force);
    for (int j = 0; j < 3; ++j) speed[i][j] += tmp[j];
    speed[i] *= 1 - F;

    {
      double a = abs(speed[i]);
      if (a > 0.4) speed[i] *= 0.4 / a;
      // if (a > d) speed[i] *= d / a;
    }

    //double a = abs(speed[i]);
    // if (a > d) speed[i] *= d / a;
    // if (a > 10) speed[i] *= 10 / a;
  }

  #pragma omp flush

  for (int i = 0; i < V; ++i) {
    pos[i] += 0.4 * speed[i];
    if (abs(pos[i]) > R) pos[i] *= R / abs(pos[i]);
  }
  }

  free(pos_m128);

  d *= D;
  return 0.0;
}
}  // namespace simd_naive_force

//
// Spectral Layout
//
/*
namespace spectral {
void SpectralLayout() {
  int rank = min(V, 20);

  // 1/2 * (I + D^{-1} A)
  vector<vector<pair<int, float> > > cols(V);
  for (int v = 0; v < V; ++v) {
    // cols[v].push_back(make_pair(v, 1.0));
    for (size_t i = 0; i < A[v].size(); ++i) {
      int u = A[v][i];
      if (u == v) continue;
      int dv = A[v].size(), du = A[u].size();
      cols[v].push_back(make_pair(A[v][i], 1.0 / sqrt(du) / sqrt(dv)));
      //cols[v].push_back(make_pair(A[v][i], 1.0));
    }
    cols[v].push_back(make_pair(v, 1));
    sort(cols[v].begin(), cols[v].end());

    //for (int u = 0; u < V; ++u) cols[v].push_back(make_pair(u, 1));
    // cols[v].push_back(make_pair(1, 1));
    // cols[v].push_back(make_pair(V - 1, 2));
  }

  REDSVD::SMatrixXf mat;
  REDSVD::Util::convertFV2Mat(cols, mat);

  REDSVD::RedSymEigen rse(mat, rank);
  const Eigen::MatrixXf &eigen_vectors = rse.eigenVectors();
  const Eigen::VectorXf &eigen_values  = rse.eigenValues();

  for (int v = 0; v < V; ++v) {
    for (int d = 0; d < 3; ++d) {
      pos[v][d] = eigen_vectors(v, rank - d - 1) * sqrt(V) / sqrt(A[v].size());
    }
  }
}
}  // namespace spectral
*/

namespace barnes_hut {
const double D = 0.999;
const double K = 1.0;
const double B = 0.5;
const double G = 0.05;
const double F = 0.5;
const double R = 5;
static double d = 0.1;

struct BHTree {
  vec3d p;   // Position
  int m;     // Mass (number)
  double s;  // Size (width)
  BHTree *child[2][2][2];
  BHTree(vec3d p, float m) : p(p), m(m) {
    for (int xi = 0; xi < 2; ++xi) {
      for (int yi = 0; yi < 2; ++yi) {
        for (int zi = 0; zi < 2; ++zi) {
          child[xi][yi][zi] = NULL;
        }
      }
    }
  }
};

BHTree *construct(vector<vec3d> &ps, double *x, double *y, double *z, int depth) {
  if (ps.size() == 0) {
    return NULL;
  } else if (ps.size() == 1 || depth == 30) {
    BHTree *r = new BHTree(accumulate(ps.begin(), ps.end(), vec3d(0.0, 3)) / (double)ps.size(),
                           ps.size());
    assert(r->p.size() == 3);
    return r;
  } else {
    double tx[3] = {x[0], (x[0] + x[1]) / 2, x[1]};
    double ty[3] = {y[0], (y[0] + y[1]) / 2, y[1]};
    double tz[3] = {z[0], (z[0] + z[1]) / 2, z[1]};

    vector<vec3d> cps[2][2][2];
    for (size_t i = 0; i < ps.size(); ++i) {
      cps [ps[i][0] < tx[1] ? 0 : 1]
          [ps[i][1] < ty[1] ? 0 : 1]
          [ps[i][2] < tz[1] ? 0 : 1].push_back(ps[i]);
    }

    BHTree *r = new BHTree(vec3d(0.0, 3), 0);
    r->s = x[1] - x[0];

    int hoge = 0;
    for (int xi = 0; xi < 2; ++xi) {
      for (int yi = 0; yi < 2; ++yi) {
        for (int zi = 0; zi < 2; ++zi) {
          hoge += cps[xi][yi][zi].size();
          BHTree *c = construct(cps[xi][yi][zi], tx + xi, ty + yi, tz + zi, depth + 1);
          r->child[xi][yi][zi] = c;
          if (c != NULL) {
            r->p += c->p;
            r->m += c->m;
          }
        }
      }
    }
    assert(hoge == (int)ps.size());
    // ps.clear();
    r->p /= r->m;
    assert(r->p.size() == 3);
    return r;
  }
}

void destruct(BHTree *t) {
  if (t == NULL) return;

  for (int xi = 0; xi < 2; ++xi) {
    for (int yi = 0; yi < 2; ++yi) {
      for (int zi = 0; zi < 2; ++zi) {
        destruct(t->child[xi][yi][zi]);
      }
    }
  }
  delete t;
}

size_t visited = 0, added = 0;

vec3d query(BHTree *t, vec3d p) {
  static const double kTheta = 0.5;
  static const double kTheta2 = kTheta * kTheta;

  ++visited;

  if (t == NULL) return vec3d(0.0, 3);

  if (t->m == 1 || t->s * t->s / norm((vec3d)(t->p - p)) < kTheta2) {
    ++added;

    vec3d d = t->p - p;
    double a = abs(d);
    if (a < EPS) {
      return vec3d(0.0, 3);  // Probably itself
    } else {
      return -d * B / (EPS + a * a * a);
    }
  } else {
    vec3d f(0.0, 3);
    for (int xi = 0; xi < 2; ++xi) {
      for (int yi = 0; yi < 2; ++yi) {
        for (int zi = 0; zi < 2; ++zi) {
          BHTree *c = t->child[xi][yi][zi];
          assert(c == NULL || c->m > 0);
          if (c) f += query(c, p);
        }
      }
    }
    return f;
  }
}

void debug_print(BHTree *t, int depth = 0) {
  if (t == NULL) return;

  if (t->m == 1) {
    printf("%*sLEAF\n", depth, "");
    return;
  }
  for (int xi = 0; xi < 2; ++xi) {
    for (int yi = 0; yi < 2; ++yi) {
      for (int zi = 0; zi < 2; ++zi) {
        BHTree *c = t->child[xi][yi][zi];
        printf("%*s%d%d%d: %d\n", depth, "", xi, yi, zi, c == NULL ? 0 : c->m);
      }
    }
  }

  for (int xi = 0; xi < 2; ++xi) {
    for (int yi = 0; yi < 2; ++yi) {
      for (int zi = 0; zi < 2; ++zi) {
        BHTree *c = t->child[xi][yi][zi];
        debug_print(c, depth + 1);
      }
    }
  }
  ++visited;
}

double Step() {
  BHTree *root;
  benchmark("construct") {
    vector<vec3d> ps(V, vec3d(0.0, 3));
    double max_XYZ = 0.0;
    for (int i = 0; i < V; ++i) {
      ps[i] = pos[i];
      max_XYZ = max(max(max_XYZ, fabs(ps[i][0])), max(fabs(ps[i][1]), fabs(ps[i][2])));
    }
    double xyzs[2] = {-max_XYZ, max_XYZ};
    root = construct(ps, xyzs, xyzs, xyzs, 0);
  }
  // debug_print(root);

  added = visited = 0;

  benchmark("querying") {
#pragma omp flush

// #pragma omp parallel for
  for (int i = 0; i < V; ++i) {
    vec3d force(0.0, 3);

    // sort(A[i].begin(), A[i].end());

    // for (int k = 0; k < V; ++k) {
    //   int j = k;
    //   if (i == j) continue;
    //   vec3d d = pos[j] - pos[i];
    //   double a = abs(d);
    //   force -= d * (B / E.size() / (EPS + a * a * a) * A[j].size());
    // }
    force += query(root, pos[i]) / (double)E.size();
    // printf("%f %f %f (%d)\n", force[0], force[1], force[2], visited);

    for (size_t k = 0; k < A[i].size(); ++k) {
      int j = A[i][k];
      if (i == j) continue;
      vec3d d = pos[j] - pos[i];
      force += d * K;
    }

    force -= pos[i] * G;
    speed[i] += force / (double)A[i].size();
    speed[i] *= 1 - F;

    //double a = abs(speed[i]);
    // if (a > d) speed[i] *= d / a;
    // if (a > 10) speed[i] *= 10 / a;
  }
  }

#pragma omp flush

  for (int i = 0; i < V; ++i) {
    pos[i] += 0.1 * speed[i];
    if (abs(pos[i]) > R) pos[i] *= R / abs(pos[i]);
  }

  printf("average: visited=%f(%f), added=%f(%f)\n",
         visited / (double)V, visited / (double)V / V,
         added / (double)V, added / (double)V / V);

  destruct(root);

  d *= D;
  return 0.0;
}
}  // nemespace barnes_hut


//
// Color (-> color.cc ?)
//
namespace coloring {
inline vec3d MakeVec3d(double a, double b, double c) {
  vec3d v(0.0, 3);
  v[0] = a;
  v[1] = b;
  v[2] = c;
  return v;
}

inline vec3d HSVtoRGB(double h, double s, double v) {
  int i = floor(h * 6.0);
  double f = h * 6.0 - i;
  if (i % 2 == 0) f = 1.0 - f;
  double m = v * (1.0 - s), n = v * (1.0 - s * f);
  switch (i % 6) {
    case 0: return MakeVec3d(v, n, m);
    case 1: return MakeVec3d(n, v, m);
    case 2: return MakeVec3d(m, v, n);
    case 3: return MakeVec3d(m, n, v);
    case 4: return MakeVec3d(n, m, v);
    case 5: return MakeVec3d(v, m, n);
  }
  assert(false);
}

inline vec3d HSVtoRGB(vec3d &hsv) {
  return HSVtoRGB(hsv[0], hsv[1], hsv[2]);
}

void Color() {
  int max_deg = 0;
  for (int v = 0; v < V; ++v) max_deg = max(max_deg, (int)A[v].size());

  vertex_alpha.resize(V);
  for (int v = 0; v < V; ++v) {
    vertex_alpha[v] = 0.5 + 0.5 * A[v].size() / (double)max_deg;
    // log(adj[v].size()) / log(max_deg);
  }

  color.resize(V, vec3d(0.0, 3));

  if (!cmdline_parser.get<string>("color_file").empty()) {
    ifstream ifs(cmdline_parser.get<string>("color_file").c_str());
    vector<int> gs(V);
    for (int v = 0; v < V; ++v) ifs >> gs[v];
    int G = max(1, *max_element(gs.begin(), gs.end()));
    for (int v = 0; v < V; ++v) {
      if (gs[v] == 0) {
        color[v] = HSVtoRGB(0.0, 0.0, 0.0);
        vertex_alpha[v] = 0.1;
      } else {
        color[v] = HSVtoRGB(gs[v] / (double)G, 1.0, 0.9);
      }
    }
  } else {
    for (int v = 0; v < V; ++v) {
      color[v] = HSVtoRGB(A[v].size() / (double)max_deg, 1.0, 1.0);
      //color[v] = HSVtoRGB(0.3, A[v].size() / (double)max_deg, 0.5);
      vertex_alpha[v] = log(A[v].size()) / log(max_deg);
    }
  }
}
}

//
// Entry points
//
void InitLayout() {
  speed.resize(V, vec3d(0.0, 3));
  pos.resize(V, vec3d(0.0, 3));

  srand(12);

  for (int v = 0; v < V; ++v) {
    for (int i = 0; i < 3; ++i) {
      pos[v][i] = (rand() / (double)RAND_MAX - 0.5);  // * (max_deg - A[v].size()) / (ndouble)max_deg;
    }
  }

  // spectral::SpectralLayout();
  coloring::Color();

  for (int i = 0; i < 30; ++i) {
    naive_force::Step3();
  }
}

void *LayoutThread(void *arg) {

  for (int i = 0; i < 300; ++i) {
    //naive_force::Step3();
  }
  for (int i = 0; i < V; ++i) {
    for (int d = 0; d < 3; ++d) {
      pos[i][d] += rand() / (double)RAND_MAX * EPS;
    }
  }

  bool is_2d = cmdline_parser.get<bool>("layout_2d");
  if (is_2d) {
    for (int v = 0; v < V; ++v) pos[v][2] = 0.0;
    for (int v = 0; v < V; ++v) speed[v][2] = 0.0;
  }

  for (int iter = 0; iter < 100; ++iter) {
    //naive_force::Step();
    // naive_force::Step2();
    // barnes_hut::Step();
    // naive_force::Step3();

    simd_naive_force::SIMDStep();
    usleep(1);
  }
  return NULL;
}
}  // namespace layout
