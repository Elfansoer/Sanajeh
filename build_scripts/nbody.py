# -*- coding: utf-8 -*-
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir + '/src')


from sanajeh import PyCompiler

compiler: PyCompiler = PyCompiler("examples/nbody/nbody.py", "nbody")
print("compile")
compiler.compile()
#compiler.printCppAndHpp()
#compiler.printCdef()
print("build")
compiler.build()
print("finish")