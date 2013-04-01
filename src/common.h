#pragma once

#include <GL/freeglut.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdarg.h>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <valarray>
#include <istream>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <limits>
#include <cstdlib>
#include <cassert>
#include <numeric>
#include "cmdline.h"

typedef std::valarray<double> vec3d;

const double EPS = 1E-9;

struct __bench__ {
  double start;
  char msg[100];
  __bench__(const char* format, ...)
  __attribute__((format(printf, 2, 3)))
  {
    va_list args;
    va_start(args, format);
    vsnprintf(msg, sizeof(msg), format, args);
    va_end(args);

    start = sec();
  }
  ~__bench__() {
    fprintf(stderr, "%s: %.6f sec\n", msg, sec() - start);
  }
  double sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }
  operator bool() { return false; }
};

#define benchmark(...) if(__bench__ __b__ = __bench__(__VA_ARGS__));else

//
// common.cc
//
extern int V;
extern std::vector<std::pair<int, int> > E;
extern std::vector<std::vector<int> > A;
extern std::vector<vec3d> pos;
extern std::vector<vec3d> color;
extern std::vector<double> vertex_alpha;  // [0, 1] (multiplied by |edge_alpha|)

extern cmdline::parser cmdline_parser;

double norm(const vec3d&);
double abs(const vec3d&);
double dot(const vec3d&, const vec3d&);
vec3d normalize(const vec3d&);

//
// layout.cc
//
namespace layout {
void InitLayout();
void *LayoutThread(void *arg);
}

//
// graph.cc
//
namespace graph {
void LoadGraph(const char *file);
}

//
// display.cc
//
namespace display {
extern vec3d camera_direction, camera_axis_up, camera_axis_right;
extern double camera_distance;

extern int head, pitch;
extern double edge_alpha;

void InitDisplay();
void Display();
void Resize(int, int);
}

//
// interaction.cc
//
namespace interaction {
namespace mouse {
void Mouse(int, int, int, int);
void Motion(int, int);
void MouseWheel(int, int, int, int);
}
namespace keyboard {
void Keyboard(unsigned char, int, int);
}
}
