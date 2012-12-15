#include "common.h"

using namespace std;

namespace display {
int head = 0, pitch = 0;
double edge_alpha = 0.3;

vec3d camera_direction, camera_axis_up, camera_axis_right;
double camera_distance = 10.0;

void InitDisplay() {
  camera_direction.resize(3);
  camera_direction[2] = 1.0;

  camera_axis_up.resize(3);
  camera_axis_up[1] = 1.0;

  camera_axis_right.resize(3);
  camera_axis_right[0] = 1.0;
}

// h, s, v \in [0.0, 1.0]
inline void glColor4d_HSV(double h, double s, double v, double a) {
  int i = floor(h * 6.0);
  double f = h * 6.0 - i;
  if (i % 2 == 0) f = 1.0 - f;
  double m = v * (1.0 - s), n = v * (1.0 - s * f);
  switch (i % 6) {
    case 0: glColor4d(v, n, m, a); return;
    case 1: glColor4d(n, v, m, a); return;
    case 2: glColor4d(m, v, n, a); return;
    case 3: glColor4d(m, n, v, a); return;
    case 4: glColor4d(n, m, v, a); return;
    case 5: glColor4d(v, m, n, a); return;
  }
}

void Display() {
  glClear(GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(30.0, 1.0, 1.0, 100.0);
  glMatrixMode(GL_MODELVIEW);


  // glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  {
    vec3d cp = camera_direction * camera_distance;  // Camera position
    gluLookAt(cp[0], cp[1], cp[2], 0.0, 0.0, 0.0, camera_axis_up[0], camera_axis_up[1], camera_axis_up[2]);
  }
  glRotatef(head, 0, 1, 0);
  glRotatef(pitch, 1, 0, 0);
  //gluLookAt(0.0, 0.0, distance, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  //glRotate(bank, 0, 0, 1);
  // glMatrixMode(GL_MODELVIEW);

  /*
  // Axis
  glColor3d(1.0, 0.0, 0.0);
  glBegin(GL_LINES);
  glVertex3d(0, 0, 0); glVertex3d(1, 0, 0);
  glVertex3d(0, 0, 0); glVertex3d(0, 1, 0);
  glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
  glEnd();
  */

  //glColor3d(1.0, 1.0, 1.0);
  //glColor3d(0.5, 0.5, 0.5);
  //glColor4d(1.0, 1.0, 1.0, edge_alpha);

  // TODO: fast sorting
  vector<pair<float, int> > edge_order(E.size());
  for (size_t i = 0; i < E.size(); ++i) {
    int u = E[i].first, v = E[i].second;
    edge_order[i].first = dot(camera_direction, (pos[u] + pos[v]) / 2.0);
    edge_order[i].second = i;
  }
  sort(edge_order.begin(), edge_order.end());

  glBegin(GL_LINES);
  for (size_t i = 0; i < E.size(); ++i) {
    int u = E[edge_order[i].second].first, v = E[edge_order[i].second].second;
    //glColor4d_HSV(e / (double)E.size(), 1.0, 1.0, edge_alpha);
    glColor4d_HSV(color[u], 1.0, 1.0, edge_alpha * vertex_alpha[u]);
    glVertex3d(pos[u][0], pos[u][1], pos[u][2]);
    glColor4d_HSV(color[v], 1.0, 1.0, edge_alpha * vertex_alpha[v]);
    glVertex3d(pos[v][0], pos[v][1], pos[v][2]);
  }
  glEnd();

  /*
  glColor3d(1.0, 1.0, 1.0);
  for (int v = 0; v < V; ++v) {
    if (adj[v].size()) {
      glPushMatrix();
      glTranslatef(pos[v][0], pos[v][1], pos[v][2]);

      // glutSolidSphere(0.02 * pow(size[v], 1/3.), 2, 2);
      glPopMatrix();
    }
  }
  */

  glutSwapBuffers();
}

void Resize(int w, int h) {
  glViewport(0, 0, w, h);
  glLoadIdentity();
  gluPerspective(30.0, (double)w / (double)h, 1.0, 100.0);
  // glTranslated(-1, -0.5, -5.0);
}
}  // namespace display
