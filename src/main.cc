#include "common.h"

double get_current_time_sec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void Timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(50, Timer, 0);
}

int main(int argc, char *argv[]) {
  graph::LoadGraph(argv[1]);

  display::InitDisplay();
  layout::InitLayout();

  pthread_t th;
  pthread_create(&th, NULL, &layout::LayoutThread, NULL);

  glutInitWindowSize(640, 480);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutCreateWindow(argv[0]);
  glutDisplayFunc(display::Display);
  glutReshapeFunc(display::Resize);
  glutMouseFunc(interaction::mouse::Mouse);
  glutKeyboardFunc(interaction::keyboard::Keyboard);
  glutMotionFunc(interaction::mouse::Motion);
  glutMouseWheelFunc(interaction::mouse::MouseWheel);
  glutTimerFunc(1, Timer, 0);
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
  glutMainLoop();

  pthread_cancel(th);
  pthread_join(th, NULL);

  return 0;
}
