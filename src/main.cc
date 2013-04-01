#include "common.h"
#include "cmdline.h"

using namespace std;

double get_current_time_sec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void Timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(50, Timer, 0);
}

void InitComandlineParser(int argc, char **argv) {
  cmdline_parser.add<string>("graph",  'g', "input file", true);
  cmdline_parser.add<string>("color_file",  0, "color", false);
  cmdline_parser.add<bool>("layout_2d",  0, "3D -> 2D?", false);

  if (cmdline_parser.parse(argc, argv) == 0){
    cerr << cmdline_parser.error() << endl
         << cmdline_parser.usage() << endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  InitComandlineParser(argc, argv);
  graph::LoadGraph(cmdline_parser.get<string>("graph").c_str());

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
