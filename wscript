# -*- python -*-

APPNAME= 'graph_draw'
VERSION= '0.0.1'

top = '.'
out = 'bin'

def options(opt):
  opt.load('compiler_cxx')

def configure(conf):
  conf.load('compiler_cxx')
  conf.env.append_value('CXXFLAGS' , ['-O3', '-fopenmp', '-Wall', '-Wextra', '-g', '-msse4.1'])
  conf.env.append_value('LINKFLAGS', ['-O3', '-fopenmp', '-Wall', '-Wextra', '-g'])
  conf.check_cxx(lib = ['glut', 'pthread', 'GLU', 'GL'], uselib_store = 'libs')

def build(bld):
  bld.recurse('src')
