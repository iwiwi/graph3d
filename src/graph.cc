#include "common.h"

using namespace std;

namespace graph {
int LoadEdgeList(std::istream &in,
                 std::vector<std::pair<int, int> > *edge_list) {
  in.sync_with_stdio(false);

  string first_line;
  getline(in, first_line);

  set<pair<int, int> > edges;

  int num_v;
  size_t num_e;
  if (2 == sscanf(first_line.c_str(), "*Vertices %d *Edges %lu", &num_v, &num_e)) {
    // Files with the format like |dataset_sigmod10/real_data/*.pjk|
    for (size_t e = 0; e < num_e; ++e) {
      int s, t;
      in >> s >> t;
      edges.insert(make_pair(min(s, t), max(s, t)));
    }
  } else if (2 == sscanf(first_line.c_str(), "p sp %d %lu", &num_v, &num_e)) {
    // DIMACS format
    for (size_t i = 0; i < num_e; ++i) {
      char a;
      int s, t;
      in >> a >> s >> t;
      edges.insert(make_pair(min(s, t), max(s, t)));
    }
  } else if (2 == sscanf(first_line.c_str(), "%d%lu", &num_v, &num_e)) {
    // fprintf(stderr, "format: edge lists\n");
    // V E
    // v_1 w_1
    // v_2 w_2
    // ...
    size_t lines = 0;
    int max_v = 0;
    for (int s, t; in >> s >> t; lines++) {
      edges.insert(make_pair(s, t));
      max_v = max(max_v, max(s, t));
    }
    if (lines == num_e) {
      // OK, the first line was "V E"
      // fprintf(stderr, "format: oops, actually the first line was V E\n");
    } else {
      // !!, the first line was actually "v_0 w_0"
      edges.insert(make_pair(num_v, (int)num_e));
      num_v = max_v + 1;
    }
  } else if (1 == sscanf(first_line.c_str(), "%d", &num_v)) {
    int s, t;
    while (in >> s >> t) {
      edges.insert(make_pair(min(s, t), max(s, t)));
    }
  } else {
    puts("!????");
    exit(1);
  }

  printf("V = %d\n", num_v);

  // Convert from 1-index to 0-index
  int min_label = numeric_limits<int>::max();
  int max_label = numeric_limits<int>::min();
  for (set<pair<int, int> >::iterator ite = edges.begin();
       ite != edges.end(); ++ite) {
    min_label = min(min_label, min(ite->first, ite->second));
    max_label = max(max_label, max(ite->first, ite->second));
  }

  const bool one_index = max_label == num_v;

  printf("E = %d\n", (int)edges.size());

  edge_list->clear();
  for (set<pair<int, int> >::iterator ite = edges.begin();
         ite != edges.end(); ++ite) {
    const int s = ite->first  + (one_index ? -1 : 0);
    const int t = ite->second + (one_index ? -1 : 0);
    edge_list->push_back(make_pair(s, t));
  }

  return num_v;
}

int LoadEdgeList(const char *file,
                 std::vector<std::pair<int, int> > *edge_list) {
  ifstream ifs(file);
  assert(ifs);
  return LoadEdgeList(ifs, edge_list);
}

void LoadGraph(const char *file) {
  V = LoadEdgeList(file, &E);

  A.resize(V);
  for (int i = 0; i < (int)E.size(); ++i) {
    A[E[i].first].push_back(E[i].second);
    A[E[i].second].push_back(E[i].first);
  }
}
}  // graph
