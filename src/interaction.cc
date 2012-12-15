#include "common.h"

using namespace std;

namespace {
// Quaternion
struct quat_t {
  float w, x, y, z;
  quat_t(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}
};

quat_t operator*(const quat_t &a, const quat_t &b) {
  return quat_t(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

vec3d rotate(vec3d src, vec3d axis, double th) {
  quat_t p(0, src[0], src[1], src[2]);
  double si = sin(th / 2), co = cos(th / 2);
  quat_t q(co, +axis[0] * si, +axis[1] * si, +axis[2] * si);
  quat_t r(co, -axis[0] * si, -axis[1] * si, -axis[2] * si);

  quat_t t = r * p * q;
  vec3d dst(0.0, 3);
  dst[0] = t.x;
  dst[1] = t.y;
  dst[2] = t.z;
  return dst;
}
}  // namespace

namespace interaction {
namespace mouse {
int state_left = GLUT_UP, state_right = GLUT_UP;
int last_x, last_y;

void Mouse(int button, int state, int x, int y) {
  if (button == GLUT_LEFT_BUTTON) {
    state_left = state;
  } else if (button == GLUT_RIGHT_BUTTON) {
    state_right = state;
  }
  last_x = x;
  last_y = y;
  glutPostRedisplay();
}

void Motion(int x, int y) {
  if (state_left == GLUT_DOWN) {
    {
      vec3d axis_up = display::camera_axis_up;
      double th_x = (x - last_x) / 100.0;
      display::camera_direction  = rotate(display::camera_direction , axis_up, th_x);
      display::camera_axis_up    = rotate(display::camera_axis_up   , axis_up, th_x);
      display::camera_axis_right = rotate(display::camera_axis_right, axis_up, th_x);
    }
    {
      vec3d axis_right = display::camera_axis_right;
      double th_y = (y - last_y) / 100.0;
      display::camera_direction  = rotate(display::camera_direction , axis_right, th_y);
      display::camera_axis_up    = rotate(display::camera_axis_up   , axis_right, th_y);
      display::camera_axis_right = rotate(display::camera_axis_right, axis_right, th_y);
    }

    display::camera_direction  = normalize(display::camera_direction);
    display::camera_axis_up    = normalize(display::camera_axis_up);
    display::camera_axis_right = normalize(display::camera_axis_right);
  } else if (state_right == GLUT_DOWN) {
    display::camera_distance += 0.1 * (x - last_x);
  }

  last_x = x;
  last_y = y;
  glutPostRedisplay();
}

void MouseWheel(int wheel_number, int direction, int x, int y) {
  puts("!?");
  // if (direction == 1) {
  //   display::distance /= 1.1;
  // } else {
  //   display::distance *= 1.1;
  // }
  glutPostRedisplay();
}
}  // namespace mouse

namespace keyboard {
void Keyboard(unsigned char key, int x, int y) {
  if (key == 'a') display::edge_alpha = min(1.0, display::edge_alpha * 1.1);
  if (key == 'z') display::edge_alpha = max(0.0, display::edge_alpha * 0.9);
  glutPostRedisplay();
}
}  // namespace keyboard
}  // namespace interaction
