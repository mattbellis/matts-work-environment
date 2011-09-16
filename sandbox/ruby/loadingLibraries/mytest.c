#include "ruby.h"

static VALUE method_test1(VALUE self) 
{
  int x = 10;
  return INT2NUM(x);
}

void Init_example() 
{
  VALUE Example = rb_define_module("Example");
  rb_define_method(Example, "test1", method_test1, 0);
}


