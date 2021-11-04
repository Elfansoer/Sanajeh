# -*- coding: utf-8 -*-
# Mark all device functions

import ast
import copy
import hashlib
import os, sys

import astunparse

import type_converter

from config import INDENT

from call_graph import CallGraph, ClassNode, FunctionNode, VariableNode
from gen_cppast import GenCppAstVisitor
import build_cpp

def pprint(node):
    print(astunparse.unparse(node))

def get_annotation(ann):
    if ann is not None:
        if type(ann) is ast.Name:
            return ann.id, None
        elif type(ann) is ast.Attribute:
            return ann.attr, None
        # Nested list is not supported yet
        elif type(ann) is ast.Subscript and ann.value.id == "list":
            return ann.value.id, get_annotation(ann.slice.value)
    return None, None  

# Generate python call graph
class GenPyCallGraphVisitor(ast.NodeVisitor):

    def __init__(self):
        self.__root = CallGraph('root')
        self.__pp = Preprocessor(self.__root)
        self.__node_path = [self.__root]
        self.__current_node = None
        self.__variables = {}

    @property
    def root(self):
        return self.__root

    # JUST FOR DEBUG
    @property
    def node_path(self):
        return self.__node_path

    # JUST FOR DEBUG
    @property
    def variables(self):
        return self.__variables

    def visit(self, node):
        self.__current_node = self.__node_path[-1]
        super(GenPyCallGraphVisitor, self).visit(node)

    def expand_type(self, ftype):
        result = []
        for class_node in self.__root.declared_classes:
            if class_node.name == ftype.v_type:
                for nested_field in class_node.declared_fields:
                    result.extend(self.expand_type(nested_field))
                return result
        result.append(ftype)
        return result

    def expand_fields(self):
        for class_node in self.__root.declared_classes:
            for field in class_node.declared_fields:
                if field.name.split("_")[-1] != "ref":
                    class_node.expanded_fields[field.name] = self.expand_type(field)
                else:
                    class_node.expanded_fields[field.name] = [field]

    def visit_Module(self, node):
        self.generic_visit(node)
        self.expand_fields()

    # Create nodes for all classes declared
    def visit_ClassDef(self, node):
        if type(self.__current_node) is not CallGraph:
            print("Doesn't support nested classes", file=sys.stderr)
            sys.exit(1)
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name)
        if class_node is not None:
            # Program shouldn't come to here, which means a class is defined twice
            print("The class {} is defined twice.".format(class_name), file=sys.stderr)
            sys.exit(1)
        base = None
        # todo does not supports multiple inheritance
        if len(node.bases) > 0:
            base = node.bases[0].id
        class_node = ClassNode(node.name, base)
        self.__current_node.declared_classes.add(class_node)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        func_name = node.name           
        if type(self.__current_node) is not CallGraph and type(self.__current_node) is not ClassNode \
        and self.__current_node.name != "main":
            print("Doesn't support nested functions", file=sys.stderr)
            sys.exit(1)
        func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
        if func_node is not None:
            # Program shouldn't come to here, which means a function is defined twice
            print("The function {} is defined twice.".format(func_name), file=sys.stderr)
            sys.exit(1)
        ret_type = None
        if type(self.__current_node) is ClassNode and func_name == "__init__":
            ret_type = self.__current_node.name
        elif hasattr(node.returns, "id"):
            ret_type = node.returns.id
        func_node = FunctionNode(func_name, self.__current_node.name, ret_type)
        self.__current_node.declared_functions.add(func_node)
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Add arguments to the environment
    def visit_arguments(self, node):
        if type(self.__current_node) is not FunctionNode:
            print('Unexpected node "{}"'.format(self.__current_node.name), file=sys.stderr)
            sys.exit(1)
        for arg in node.args:
            var_type = None
            element_type = None
            if arg.arg == "self":
                continue
            if hasattr(arg, "annotation") and arg.annotation is not None:
                # annotation = arg.annotation.id
                var_type, element_type = get_annotation(arg.annotation)
            var_node = VariableNode(arg.arg, var_type, element_type)
            self.__current_node.arguments.add(var_node)
            self.__variables.setdefault(self.__current_node.id, []).append(arg.arg)

    # Add global variables to the environment
    def visit_Global(self, node):
        for global_variable in node.names:
            self.__variables.setdefault(self.__current_node.id, []).append(global_variable)
            var_node = self.__root.GetVariableNode(global_variable, None)
            if var_node is None:
                print("The global variable {} is not existed.".format(global_variable), file=sys.stderr)
                sys.exit(1)
            self.__current_node.called_variables.add(var_node)

    # Create nodes for variables without type annotation
    def visit_Assign(self, node):
        for var in node.targets:
            var_name = None

            if type(var) is ast.Attribute:
                pass
                # var_name = var.attr
                # print(var_name, var.value.id)
                # # todo Attribute variables(self should refer to the class not in the current block),
                # # todo haven't thought about other occasions
                # if var.value.id == 'self':
                #     pass
            elif type(var) is ast.Name:
                var_name = var.id
                self.__variables.setdefault(self.__current_node.id, [])
                if var_name not in self.__variables[self.__current_node.id]:
                    var_node = VariableNode(var_name, None)
                    self.__current_node.declared_variables.add(var_node)
                    self.__variables[self.__current_node.id].append(var_name)

        self.generic_visit(node)

    # Create nodes for variables with type annotation
    def visit_AnnAssign(self, node):
        var = node.target
        ann, e_ann = get_annotation(node.annotation)
        """
        if type(node.annotation) is ast.Subscript:
            ann = node.annotation.value.id
            e_ann = node.annotation.slice.value.id
        else:
            if hasattr(node.annotation, "attr"):
                ann = node.annotation.attr
            else:
                ann = node.annotation.id
        """
        if type(var) is ast.Attribute:
            var_name = var.attr
            if hasattr(var.value, "id") and var.value.id == "self" and self.__current_node.name == "__init__":
                field_node = VariableNode(var_name, ann, e_ann)
                self.node_path[-2].declared_fields.add(field_node)
            # todo Attribute variables(self should refer to the class not in the current block),
            # todo haven't thought about other occasions
        elif type(var) is ast.Name:
            var_name = var.id
            self.__variables.setdefault(self.__current_node.id, [])
            if var_name not in self.__variables[self.__current_node.id]:
                var_node = VariableNode(var_name, ann, e_ann)
                self.__current_node.declared_variables.add(var_node)
                self.__variables[self.__current_node.id].append(var_name)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.__variables.setdefault(self.__current_node.id, [])
        if node.id in self.__variables[self.__current_node.id]:
            return
        for annotate_location_node in self.__node_path[-2::-1]:
            self.__variables.setdefault(annotate_location_node.id, [])
            if node.id in self.__variables[annotate_location_node.id]:
                var_node = annotate_location_node.GetVariableNode(node.id, None)
                if var_node is None:
                    print('Unexpected error, can not find variable "{}"', node.id, file=sys.stderr)
                    sys.exit(1)
                self.__current_node.called_variables.add(var_node)
                break

    def visit_Call(self, node):
        if hasattr(node.func, "value") and hasattr(node.func.value, "id") \
        and node.func.value.id == "random" and node.func.attr == "seed" \
        and type(self.__node_path[-2]) is ClassNode:
            self.__node_path[-2].has_random_state = True
        self.generic_visit(node)

    # mark all device data in the CallGraph
    def mark_device_data(self, node):
        self.__pp.visit(node)
        if not self.__pp.has_device_data:
            return False
        self.__root.MarkDeviceDataByClassName(self.__pp.classes)
        return True

    def build_parallel_do_cpp(self):
        return '\n\n' + '\n\n'.join(self.__pp.cpp_parallel_do_codes)

    def build_parallel_do_hpp(self):
        return '\n'.join(self.__pp.hpp_parallel_do_codes)

    def build_parallel_do_cdef(self):
        return '\n' + '\n'.join(self.__pp.cdef_parallel_do_codes)

    def build_do_all_cpp(self):
        return '\n\n'.join(self.__pp.cpp_do_all_codes)

    def build_do_all_hpp(self):
        return '\n' + '\n'.join(self.__pp.hpp_do_all_codes)

    def build_do_all_cdef(self):
        return '\n' + '\n'.join(self.__pp.cdef_do_all_codes)

    def build_parallel_new_cpp(self):
        return '\n\n' + '\n\n'.join(self.__pp.cpp_parallel_new_codes)

    def build_parallel_new_hpp(self):
        return '\n' + '\n'.join(self.__pp.hpp_parallel_new_codes)

    def build_parallel_new_cdef(self):
        return '\n' + '\n'.join(self.__pp.cdef_parallel_new_codes)

    def build_global_device_variables_init(self):
        ret = []
        for var in self.__pp.global_device_variables:
            var_node = self.__root.GetVariableNode(var, None)
            e_type = type_converter.convert(var_node.e_type)
            n = self.__pp.global_device_variables[var]
            ret.append(INDENT + "{}* host_{};\n".format(e_type, var) + \
                       INDENT + "cudaMalloc(&host_{}, sizeof({})*{});\n".format(var, e_type, n) + \
                       INDENT + "cudaMemcpyToSymbol({}, &host_{}, sizeof({}*), 0, cudaMemcpyHostToDevice);\n" \
                       .format(var, var, e_type))
        return "\n".join(ret)

    def build_global_device_variables_unit(self):
        ret = []
        for var in self.__pp.global_device_variables:
            ret.append(INDENT + "cudaFree(host_{});\n".format(var))
        return "\n".join(ret)      


# Find device class in python code and compile parallel_do expressions into c++ ones
class Preprocessor(ast.NodeVisitor):

    @property
    def cpp_parallel_do_codes(self):
        return self.__cpp_parallel_do_codes

    @property
    def hpp_parallel_do_codes(self):
        return self.__hpp_parallel_do_codes

    @property
    def cdef_parallel_do_codes(self):
        return self.__cdef_parallel_do_codes

    @property
    def cpp_parallel_new_codes(self):
        return self.__cpp_parallel_new_codes

    @property
    def hpp_parallel_new_codes(self):
        return self.__hpp_parallel_new_codes

    @property
    def cdef_parallel_new_codes(self):
        return self.__cdef_parallel_new_codes

    @property
    def cpp_do_all_codes(self):
        return self.__cpp_do_all_codes

    @property
    def hpp_do_all_codes(self):
        return self.__hpp_do_all_codes

    @property
    def cdef_do_all_codes(self):
        return self.__cdef_do_all_codes

    # Build codes for parallel_new in c++
    class ParallelNewBuilder:
        def __init__(self, class_name):
            self.__class_name = class_name  # The class of the object

        def buildCpp(self):
            parallel_new_expr = INDENT + "allocator_handle->parallel_new<{}>(object_num);\n".format(self.__class_name)
            return_expr = INDENT + "return 0;"
            return 'extern "C" int parallel_new_{}(int object_num){{\n'.format(self.__class_name) \
                   + parallel_new_expr \
                   + return_expr \
                   + "\n}"

        def buildHpp(self):
            return 'extern "C" int parallel_new_{}(int object_num);'.format(self.__class_name)

        def buildCdef(self):
            return 'int parallel_new_{}(int object_num);'.format(self.__class_name)

    # Collect information of those functions used in the parallel_do function, and build codes for that function in c++
    class ParallelDoBuilder(ast.NodeVisitor):
        def __init__(self, rt, class_name, func_class_name, func_name):
            self.__root = rt
            self.__node_path = [rt]
            self.__current_node = None
            self.__object_class_name = class_name  # The class of the object
            self.__func_class_name = func_class_name  # The class of the function executed
            self.__func_name = func_name
            self.__args = {}

        def visit(self, node):
            self.__current_node = self.__node_path[-1]
            super(Preprocessor.ParallelDoBuilder, self).visit(node)

        def visit_ClassDef(self, node):
            if node.name != self.__func_class_name:
                return
            class_name = node.name
            class_node = self.__current_node.GetClassNode(class_name)
            if class_node is None:
                # Program shouldn't come to here, which means the class does not exist
                print("The class {} is not exist.".format(class_name), file=sys.stderr)
                sys.exit(1)
            self.__node_path.append(class_node)
            self.generic_visit(node)
            self.__node_path.pop()

        def visit_FunctionDef(self, node):
            func_name = node.name
            func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
            if func_node is None:
                # Program shouldn't come to here, which means the function does not exist
                print("The function {} does not exist.".format(func_name), file=sys.stderr)
                sys.exit(1)
            if func_name != self.__func_name or self.__current_node.name != self.__func_class_name:
                return
            for arg_ in node.args.args:
                if arg_.arg == 'self':
                    continue
                self.__args[arg_.arg] = arg_.annotation.id

        def buildCpp(self):
            arg_strs = []
            for arg_ in self.__args:
                arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))
            parallel_do_expr = INDENT + "allocator_handle->parallel_do<{}, &{}::{}>({});\n".format(
                self.__object_class_name,
                self.__func_class_name,
                self.__func_name,
                ", ".join(self.__args)
            )
            return_expr = INDENT + "return 0;"

            return 'extern "C" int {}_{}_{}({}){{\n'.format(
                self.__object_class_name,
                self.__func_class_name,
                self.__func_name,
                ", ".join(arg_strs)) \
                   + parallel_do_expr \
                   + return_expr \
                   + "\n}"

        def buildHpp(self):
            arg_strs = []
            for arg_ in self.__args:
                arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))

            return 'extern "C" int {}_{}_{}({});'.format(
                self.__object_class_name,
                self.__func_class_name,
                self.__func_name,
                ",".join(arg_strs)
            )

        def buildCdef(self):
            arg_strs = []
            for arg_ in self.__args:
                # arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))
                args_strs.append("{} {}".format(type_converter.cdef_convert(self.__args[arg_]). arg_))

            return 'int {}_{}_{}({});'.format(
                self.__object_class_name,
                self.__func_class_name,
                self.__func_name,
                ",".join(arg_strs)
            )

    # Collect information of class fields and build codes do_all functions in c++
    class DoAllBuilder(ast.NodeVisitor):
        def __init__(self, rt, class_name):
            self.__root = rt
            self.__node_path = [rt]
            self.__current_node = None
            self.__class_name = class_name
            self.__field = {}

        def visit(self, node):
            self.__current_node = self.__node_path[-1]
            super(Preprocessor.DoAllBuilder, self).visit(node)

        def visit_ClassDef(self, node):
            if node.name != self.__class_name:
                return
            class_name = node.name
            class_node = self.__current_node.GetClassNode(class_name)
            if class_node is None:
                # Program shouldn't come to here, which means the class does not exist
                print("The class {} does not exist.".format(class_name), file=sys.stderr)
                sys.exit(1)
            self.__node_path.append(class_node)
            self.generic_visit(node)
            self.__node_path.pop()

        def visit_FunctionDef(self, node):
            func_name = node.name
            func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
            if func_node is None:
                # Program shouldn't come to here, which means the function does not exist
                print("The function {} does not exist.".format(func_name), file=sys.stderr)
                sys.exit(1)
            self.__node_path.append(func_node)
            self.generic_visit(node)
            self.__node_path.pop()

        def visit_AnnAssign(self, node):
            if type(self.__current_node) is FunctionNode and self.__current_node.name == "__init__":
                var = node.target.attr
                if type(node.annotation) is ast.Subscript and node.annotation.value.id == "list":
                    var_type = "list"
                else:
                    var_type = type_converter.convert_ann(node.annotation)
                self.__field[var] = var_type

        def buildCpp(self):
            fields_str = ""
            field_types_str = ""
            for i, field in enumerate(self.__field):
                field_type = self.__field[field]
                if i != len(self.__field) - 1:
                    if field_type not in ["int", "float", "bool"]:
                        # fields_str += "(int) this->{}, ".format(field)
                        fields_str += "0, ".format(field)
                        field_types_str += "{}, ".format("int")
                    else:
                        fields_str += "this->{}, ".format(field)
                        field_types_str += "{}, ".format(field_type)
                else:
                    if self.__field[field] not in ["int", "float", "bool"]:
                        fields_str += "0".format(field)
                        field_types_str += "{}".format("int") 
                    else:
                        fields_str += "this->{}".format(field)
                        field_types_str += "{}".format(field_type) 
            func_exprs = ['\n' +
                          'void {}::_do(void (*pf)({})){{\n'.format(self.__class_name, field_types_str) +
                          INDENT +
                          'pf({});\n'.format(fields_str) +
                          '}',
                          '\n' +
                          'extern "C" int {}_do_all(void (*pf)({})){{\n'.format(self.__class_name, field_types_str) +
                          INDENT +
                          'allocator_handle->template device_do<{}>(&{}::_do, pf);\n '.format(self.__class_name,
                                                                                              self.__class_name) +
                          INDENT + 'return 0;\n' +
                          '}']
            return "\n".join(func_exprs)

        def buildHpp(self):
            return 'extern "C" {}\n'.format(self.buildCdef())

        def buildCdef(self):
            field_types_str = ""
            for i, field in enumerate(self.__field):
                field_type = self.__field[field] if self.__field[field] in ["int", "bool", "float"] else "int"
                if i != len(self.__field) - 1:
                    field_types_str += "{}, ".format(field_type)
                else:
                    field_types_str += "{}".format(field_type)
            return 'int {}_do_all(void (*pf)({}));'.format(self.__class_name, field_types_str)

    def __init__(self, rt: CallGraph):
        self.__classes = []

        self.has_device_data = False
        self.__is_root = True  # the flag of whether visiting the root node of python ast
        self.__node_root = None  # the root node of python ast
        self.__cpp_parallel_do_codes = []
        self.__hpp_parallel_do_codes = []
        self.__cdef_parallel_do_codes = []
        self.__cpp_parallel_new_codes = []
        self.__hpp_parallel_new_codes = []
        self.__cdef_parallel_new_codes = []
        self.__cpp_do_all_codes = []
        self.__hpp_do_all_codes = []
        self.__cdef_do_all_codes = []
        self.__parallel_do_hashtable = []
        self.__root = rt
        self.__node_path = [rt]
        self.__current_node = None
        self.global_device_variables = {}

    def visit(self, node):
        if self.__is_root:
            self.__node_root = node
            self.__is_root = False
        self.__current_node = self.__node_path[-1]
        super(Preprocessor, self).visit(node)

    @property
    def classes(self):
        return self.__classes

    def __gen_Hash(self, lst):
        """
        Helper function, generate same hash value for tuple with same strings
            Used to prevent generate mutiple code for a same parallel_do function
        """
        m = hashlib.md5()
        for elem in lst:
            m.update(elem.encode('utf-8'))
        return m.hexdigest()

    def visit_ClassDef(self, node):
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name)
        if class_node is None:
            # Program shouldn't come to here, which means the class does not exist
            print("The class {} is not exist.".format(class_name), file=sys.stderr)
            sys.exit(1)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        func_name = node.name
        func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
        if func_node is None:
            # Program shouldn't come to here, which means the function does not exist
            print("The function {} does not exist.".format(func_name), file=sys.stderr)
            sys.exit(1)
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Analyze function calling relationships
    def visit_Call(self, node):
        # Find device classes through device code
        if type(node.func) is ast.Attribute and \
                hasattr(node.func.value, "id") and \
                node.func.value.id == "DeviceAllocator":
            if node.func.attr == 'device_class':
                self.has_device_data = True
                for cls in node.args:
                    if cls.id not in self.__classes:
                        self.__classes.append(cls.id)
                        pnb = self.ParallelNewBuilder(cls.id)
                        self.__cpp_parallel_new_codes.append(pnb.buildCpp())
                        self.__hpp_parallel_new_codes.append(pnb.buildHpp())
                        self.__cdef_parallel_new_codes.append(pnb.buildCdef())
                        dab = self.DoAllBuilder(self.__root, cls.id)
                        dab.visit(self.__node_root)
                        self.__cpp_do_all_codes.append(dab.buildCpp())
                        self.__hpp_do_all_codes.append(dab.buildHpp())
                        self.__cdef_do_all_codes.append(dab.buildCdef())
            elif node.func.attr == 'parallel_do':
                hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                if hval not in self.__parallel_do_hashtable:
                    self.__parallel_do_hashtable.append(hval)
                    pdb = self.ParallelDoBuilder(self.__root,
                                                 node.args[0].id,
                                                 node.args[1].value.id,
                                                 node.args[1].attr)
                    pdb.visit(self.__node_root)
                    self.__cpp_parallel_do_codes.append(pdb.buildCpp())
                    self.__hpp_parallel_do_codes.append(pdb.buildHpp())
                    self.__cdef_parallel_do_codes.append(pdb.buildCdef())
            elif node.func.attr == 'array_size':
                self.global_device_variables[str(node.args[0].id)] = node.args[1].n

        # Find device classes through host code
        if type(node.func) is ast.Attribute \
                and hasattr(node.func.value, "id") and \
                (node.func.value.id == "allocator" or node.func.value.id == "PyAllocator"):
            if node.func.attr == 'parallel_new':
                self.has_device_data = True
                if node.args[0].id not in self.__classes:
                    self.__classes.append(node.args[0].id)
                    pnb = self.ParallelNewBuilder(node.args[0].id)
                    self.__cpp_parallel_new_codes.append(pnb.buildCpp())
                    self.__hpp_parallel_new_codes.append(pnb.buildHpp())
                    self.__cdef_parallel_new_codes.append(pnb.buildCdef())
                    dab = self.DoAllBuilder(self.__root, node.args[0].id)
                    dab.visit(self.__node_root)
                    self.__cpp_do_all_codes.append(dab.buildCpp())
                    self.__hpp_do_all_codes.append(dab.buildHpp())
                    self.__cdef_do_all_codes.append(dab.buildCdef())

            elif node.func.attr == 'parallel_do':
                hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                if hval not in self.__parallel_do_hashtable:
                    self.__parallel_do_hashtable.append(hval)
                    pdb = self.ParallelDoBuilder(self.__root,
                                                 node.args[0].id,
                                                 node.args[1].value.id,
                                                 node.args[1].attr)
                    pdb.visit(self.__node_root)
                    self.__cpp_parallel_do_codes.append(pdb.buildCpp())
                    self.__hpp_parallel_do_codes.append(pdb.buildHpp())
                    self.__cdef_parallel_do_codes.append(pdb.buildCdef())

        func_name = None
        call_node = None
        var_type = None

        if type(node.func) is ast.Attribute:
            func_name = node.func.attr
            if type(node.func.value) is ast.Attribute:
                if hasattr(node.func.value.value, "id") \
                        and node.func.value.value.id == "self" \
                        and type(self.__node_path[-2]) is ClassNode:
                    # for var in self.__node_path[-2].declared_variables:
                    for var in self.__node_path[-2].declared_fields:
                        if var.name == node.func.value.attr:
                            var_type = var.v_type
        elif type(node.func) is ast.Name:
            func_name = node.func.id

        for parent_node in self.__node_path[::-1]:
            if type(parent_node) is FunctionNode:
                continue
            call_node = parent_node.GetFunctionNode(func_name, var_type)
            if call_node is not None:
                break
        if call_node is None:
            call_node = FunctionNode(func_name, var_type, None)
            self.__root.library_functions.add(call_node)
        self.__current_node.called_functions.add(call_node)
        self.generic_visit(node)

"""
Visitors for Python code transformers
"""
class DeviceCodeVisitor(ast.NodeTransformer):

    def __init__(self, root: CallGraph):
        self.root: CallGraph = root
        self.node_path = [self.root]

    def visit_Module(self, node):
        for node_body in node.body:
            if type(node_body) in [ast.ClassDef, ast.FunctionDef, ast.AnnAssign]:
                self.visit(node_body)
        return node

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.node_path[-1].GetClassNode(name)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.node_path.append(class_node)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            self.visit(node.args)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node

    def GetVariableNode(self, var_name):
        i = len(self.node_path) - 1
        while i >= 0:
            if type(self.node_path[i]) in [FunctionNode, CallGraph]:
                for var_node in self.node_path[i].declared_variables:
                    if var_node.name == var_name:
                        return var_node
            i -= 1
        return None

    """
    Type a possibly nested attribute
    """
    def attribute_type(self, attribute):
        rec_type = None
        if type(attribute.value) == ast.Name:
            if attribute.value.id == "self":
                rec_type = self.node_path[-2].name
            else:
                rec_type = self.node_path[-1].GetVariableType(attribute.value.id)
        elif type(attribute.value) == ast.Attribute:
            rec_type = self.attribute_type(attribute.value)
        elif type(attribute.value) == ast.Subscript:
            rec_type = self.attribute_type(attribute.value.value)
        if rec_type is not None and rec_type not in ["int", "bool", "float"]:
            if self.node_path[0].GetClassNode(rec_type) is not None:
                rec_class = self.root.GetClassNode(rec_type)
                for field in rec_class.declared_fields:
                    if field.name == attribute.attr:
                        return field.v_type
                """
                rec_class_name = self.node_path[0].GetClassNode(rec_type).name
                if attribute.attr in Checker.original[rec_class_name]:
                    return Checker.original[rec_class_name][attribute.attr]
                """
        return None 

class Searcher(DeviceCodeVisitor):
    """Find all classes that are used for fields or variables types in device codes"""

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.dev_cls = set()    # device classes
        self.sdef_cls = set()   # classes that are used for fields or variables types

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.node_path[-1].GetClassNode(name)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.dev_cls.add(name)
            self.node_path.append(class_node)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node

    def visit_AnnAssign(self, node):
        if hasattr(node.annotation, "id"):
            ann = node.annotation.id
            if ann not in type_converter.type_map:
                self.sdef_cls.add(ann)
        return node

    def visit_arg(self, node):
        if hasattr(node.annotation, "id"):
            ann = node.annotation.id
            if ann not in type_converter.type_map and ann is not None:
                self.sdef_cls.add(ann)
        return node

# DONE: turn this into nested compatible
class Normalizer(DeviceCodeVisitor):
    """
    Declare new variables to replace method calls nested inside other expressions.
    Example
      -- self.vel.add(self.force.multiply(kDt).divide(self.mass))
    is converted into
      -- __auto_v0: Vector = self.force.multiply(kDt)
      -- __auto_v1: Vector = __auto_v0.divide(self.mass)
      -- self.vel.add(__auto_v1)
    """

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.v_counter = 0  # used to count the auto generated variables
        self.last_annotation = None
        self.has_auto_variables = False
        self.built_nodes = []
        self.current_attr = None

    def visit_FunctionDef(self, node):
        self.v_counter = 0  # counter needs to be reset in every function
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if type(node.body[i]) == list:
                    for n in node.body[i]:
                        node.body.insert(i, n)
                        i += 1
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_Expr(self, node):
        ret = []
        self.last_annotation = "None"
        self.built_nodes = []
        self.current_attr = node.value
        node.value = self.visit(node.value)
        if self.built_nodes:
            for new_node in self.built_nodes:
                ret.append(new_node)
            ret.append(ast.Expr(value=self.current_attr))
            return ret
        else:
            return node

    def visit_Assign(self, node):
        ret = []
        self.last_annotation = "None"
        self.has_auto_variables = False
        self.built_nodes = []
        self.current_attr = node.value
        node.value = self.visit(node.value)
        if self.built_nodes:
            for new_node in self.built_nodes:
                ret.append(new_node)
            ret.append(ast.Assign(targets=node.targets, value=self.current_attr))
            return ret
        else:
            return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return node
        ret = []
        self.last_annotation = "None"
        self.has_auto_variables = False
        self.built_nodes = []
        self.current_attr = node.value
        node.value = self.visit(node.value)
        if self.built_nodes:
            for new_node in self.built_nodes:
                ret.append(new_node)
            ret.append(ast.AnnAssign(
                annotation=node.annotation, simple=node.simple, 
                target=node.target, value=self.current_attr))
            return ret
        else:
            return node

    def visit_Name(self, node):
        if node.id == "self":
            self.last_annotation = self.node_path[-2].name
        elif self.GetVariableNode(node.id) is not None:
            self.last_annotation = self.GetVariableNode(node.id).v_type
        self.current_attr = node
        return node
        
    def visit_Attribute(self, node):
        self.visit(node.value)
        if type(self.current_attr) == ast.Call and self.has_auto_variables:
            new_node = ast.AnnAssign(
                target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                value=self.current_attr, simple=1, annotation=ast.Name(id=self.last_annotation, ctx=ast.Load()))
            self.current_attr = ast.Attribute(
                value=ast.Name(id="__auto_v" + str(self.v_counter), ctx=node.value.ctx if hasattr(node.value, "ctx") else None),
                attr=node.attr, ctx=node.ctx)
            self.built_nodes.append(new_node)
            self.v_counter += 1
        else:
            self.current_attr = ast.Attribute(value=self.current_attr, attr=node.attr, ctx=node.ctx)
        class_node = self.root.GetClassNode(self.last_annotation)
        if class_node is not None:
            for field in class_node.declared_fields:
                if field.name == node.attr:
                    self.last_annotation = field.v_type
        return node

    def visit_Subscript(self, node):
        self.current_attr = node.slice.value
        self.visit(node.slice.value)
        new_index = ast.Index(self.current_attr)
        if type(node.value) == ast.Name:
            var_node = self.GetVariableNode(node.value.id)
            if var_node is not None:
                self.last_annotation = var_node.e_type[0]
            self.current_attr = node
        else:
            self.visit(node.value.value)
            if type(node.value.value) == ast.Call and self.has_auto_variables:
                new_node = ast.AnnAssign(
                    target=ast.Name(id="__auto_v"+ str(self.v_counter), ctx=ast.Load()),
                    value=self.current_attr, simple=1, annotation=ast.Name(id=self.last_annotation, ctx=ast.Load()))
                self.built_nodes.append(new_node)
                self.current_attr = ast.Subscript(
                    value=ast.Attribute(
                        value=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                        attr=node.value.attr, ctx=node.value.ctx),
                    slice=new_index, ctx=node.ctx)
            else:
                self.current_attr = ast.Subscript(
                    value=ast.Attribute(value=self.current_attr, attr=node.value.attr, ctx=node.value.ctx),
                    slice=new_index, ctx=node.ctx)
            class_node = self.root.GetClassNode(self.last_annotation)
            if class_node is not None:
                for field in class_node.declared_fields:
                    if field.name == node.value.attr:
                        self.last_annotation = field.e_type[0]
        return node

    def visit_Call(self, node):
        new_args = []
        for arg in node.args:
            self.current_attr = arg
            self.visit(arg)
            if type(arg) == ast.Call:
                if hasattr(arg.func.value, "id") and arg.func.value.id == "random" \
                and arg.func.attr in ["getrandbits", "uniform"]:
                    new_args.append(arg)
                else:
                    new_node = ast.AnnAssign(
                        target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                        value=self.current_attr, simple=1, annotation=ast.Name(id=self.last_annotation, ctx=ast.Load()))
                    self.built_nodes.append(new_node)
                    new_args.append(ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()))
                    self.v_counter += 1
            else:
                new_args.append(self.current_attr)
        if type(node.func) == ast.Name:
            for class_func in self.node_path[-2].declared_functions:
                if class_func.name == node.func.id:
                    self.last_annotation = class_func.ret_type
            for class_node in self.root.declared_classes:
                if class_node.name == node.func.id:
                    self.last_annotation = class_node.name
            self.current_attr = node
        elif hasattr(node.func.value, "id") and node.func.value.id == "random" \
        and node.func.attr in ["getrandbits", "uniform"]:
            self.current_attr = node
        else:
            if hasattr(node.func.value, "func") and hasattr(node.func.value.func, "id") \
            and node.func.value.func.id == "super":
                return node
            self.visit(node.func.value)
            if type(node.func.value) == ast.Call:
                new_node = ast.AnnAssign(
                    target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                    value=self.current_attr, simple=1, annotation=ast.Name(id=self.last_annotation, ctx=ast.Load()))
                self.built_nodes.append(new_node)
                self.current_attr = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                        attr=node.func.attr, ctx=node.func.ctx),
                    args=new_args, keywords=node.func.value.keywords)
                self.v_counter += 1
            else:
                self.current_attr = ast.Call(
                    func=ast.Attribute(value=self.current_attr, attr=node.func.attr, ctx=node.func.ctx),
                    args=new_args, keywords=node.keywords)
            class_node = self.root.GetClassNode(self.last_annotation)
            if class_node is not None:
                for class_func in class_node.declared_functions:
                    if class_func.name == node.func.attr:
                        self.last_annotation = class_func.ret_type
        return node

class FunctionBodyGenerator(ast.NodeTransformer):
    """Generate new ast nodes which are used by the Inliner to inline functions"""

    def __init__(self, node, caller_type):
        self.func_name = None
        self.caller = None
        self.args = []
        self.func_args = []
        self.new_ast_nodes = []
        self.node = copy.deepcopy(node)
        self.caller_type = caller_type

    def visit_Module(self, node):
        for x in node.body:
            if type(x) == ast.ClassDef:
                self.visit(x)
        return node

    def visit_ClassDef(self, node):
        if node.name != self.caller_type:
            return node
        else:
            node.body = [self.visit(x) for x in node.body]
        return node

    def visit_FunctionDef(self, node):
        if node.name == self.func_name:
            #  check if argument number is correct
            if len(node.args.args) - 1 != len(self.args):
                print("Invalid argument number.", file=sys.stderr)
                sys.exit(1)
            for arg in node.args.args:
                if arg.arg != "self":
                    self.func_args.append(arg.arg)
            for x in node.body:
                self.new_ast_nodes.append(self.visit(x))
        return node

    def visit_Name(self, node):
        if node.id == "self":
            return self.caller
        for i in range(len(self.func_args)):
            if node.id == self.func_args[i]:
                return self.args[i]
        return node

    def GetTransformedNodes(self, caller, func_name, args):
        """Traverse the ast node given, find the target function and return the transformed implementation"""
        self.caller = copy.deepcopy(caller)
        self.func_name = func_name
        self.args = copy.deepcopy(args)
        self.visit(self.node)

# TODO: turn this into nested compatible
class Inliner(DeviceCodeVisitor):
    """
    Replace function call on non-device classes with the specific implementations
    Example
      -- __auto_v0: Vector = self.force.multiply(kDt)
    is converted into
      -- __auto_v0: Vector = Vector((self.force.x * kDt), (self.force.y * kDt))

    TODO: Inlined variable declarations are not uniquely renamed
    """
    def __init__(self, root: CallGraph, node, sdef_cls):
        super().__init__(root)
        self.node = node
        self.sdef_cls = sdef_cls

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if type(node.body[i]) == list:
                    for n in node.body[i]:
                        node.body.insert(i, n)
                        i += 1
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_Expr(self, node):
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Return:
                    continue
                ret.append(v)
            return ret
        else:
            return node

    def visit_Assign(self, node):
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Return:
                    v = ast.Assign(targets=node.targets, value=v.value)
                ret.append(v)
            return ret
        else:
            return node

    def visit_AnnAssign(self, node):
        if node.value is None or get_annotation(node.annotation)[0] == "list":
            return node
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Return:
                    v = ast.AnnAssign(annotation=node.annotation, simple=node.simple, target=node.target, value=v.value)
                ret.append(v)
            return ret
        else:
            return node
        
    def visit_Call(self, node):
        if type(node.func) == ast.Attribute:
            if type(node.func.value) == ast.Attribute:
                if type(node.func.value.value) == ast.Name and node.func.value.attr.split("_")[-1] != "ref":
                    if node.func.value.value.id == "self" and type(self.node_path[-2]) is ClassNode:
                        caller_type = None
                        for field in self.node_path[-2].declared_fields:
                            if field.name == node.func.value.attr:
                                caller_type = field.v_type
                        if caller_type in self.sdef_cls:
                            func_body_gen = FunctionBodyGenerator(self.node, caller_type)
                            func_body_gen.GetTransformedNodes(node.func.value,
                                                            node.func.attr,
                                                            node.args)
                            if len(func_body_gen.new_ast_nodes) != 0:
                                return func_body_gen.new_ast_nodes
                    else:
                        caller_type = self.node_path[-1].GetVariableType(node.func.value.value.id)
                        var_class = self.node_path[0].GetClassNode(caller_type)   
                        var_type = None 
                        for field in var_class.declared_fields:
                            if field.name == node.func.value.attr:
                                var_type = field.v_type
                        if var_type in self.sdef_cls:
                            func_body_gen = FunctionBodyGenerator(self.node, var_type)
                            func_body_gen.GetTransformedNodes(node.func.value,
                                                            node.func.attr,
                                                            node.args)
                            if len(func_body_gen.new_ast_nodes) != 0:
                                return func_body_gen.new_ast_nodes
            elif type(node.func.value) == ast.Name:
                if node.func.value.id.split("_")[-1] != "ref":
                    caller_type = self.node_path[-1].GetVariableType(node.func.value.id)
                    if caller_type in self.sdef_cls:
                        func_body_gen = FunctionBodyGenerator(self.node, caller_type)
                        func_body_gen.GetTransformedNodes(node.func.value,
                                                        node.func.attr,
                                                        node.args)
                        if len(func_body_gen.new_ast_nodes) != 0:
                            return func_body_gen.new_ast_nodes
        return node

# TODO: turn this into nested compatible
class Eliminator(DeviceCodeVisitor):
    """
    ISSUE: 
    1. The following line causes problem, passes when there is no type annotation
    -- other.force: Vector = new_force
    // Solved by removing the part using var_dict in visit_AnnAssign //
    2. Reference type's nested fields are not expanded
    -- other.force = new_force
       is not converted into
    -- other.force.x = new_force.x
    -- other.force.y = new_force.y
    // Solved by hijacking the else in visit_AnnAssign and visit_Assign
    3. Assigning method call to attributes get annotations added although unnecessary
    -- m.pos = Vector((__auto_v3.x / 2), (__auto_v3.y / 2))
       is converted into
    -- m.pos.x: float = (__auto_v3.x / 2)
    -- m.pos.y: float = (__auto_v3.y / 2)
    // Solved by removing annotation during field synthesizing
    """
    def __init__(self, root: CallGraph, node, sdef_cls):
        super().__init__(root)
        self.node = node
        self.sdef_cls = sdef_cls

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            # self.var_dict = {}
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if type(node.body[i]) == list:
                    for n in node.body[i]:
                        node.body.insert(i, n)
                        i += 1
                    del node.body[i]
                elif node.body[i] is None:
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        """
        No inlining for reference type
        """
        if (type(node.target) == ast.Attribute and node.target.attr.split("_")[-1] == "ref") \
        or (type(node.target) == ast.Name and node.target.id.split("_")[-1] == "ref"):
            return node
        if hasattr(node.annotation, "id") and node.annotation.id in self.sdef_cls and node.value is not None:
            if type(node.value) == ast.Call and type(node.value.func) == ast.Name:
                if node.value.func.id in self.sdef_cls:
                    func_body_gen = FunctionBodyGenerator(self.node, node.value.func.id)
                    func_body_gen.GetTransformedNodes(node.target,
                                                      "__init__",
                                                      node.value.args)
                    if len(func_body_gen.new_ast_nodes) != 0:
                        return func_body_gen.new_ast_nodes
            else:
                """
                Deal with assignment by non-method/constructor calls.
                """
                if get_annotation(node.annotation)[0] == "list":
                    return node
                expand_class = self.node_path[0].GetClassNode(node.annotation.id)
                new_nodes = []
                for nested_field_name, nested_fields in expand_class.expanded_fields.items():
                    for nested_field in nested_fields:
                        ctx = node.value.ctx if hasattr(node.value, "ctx") else None
                        new_node = ast.AnnAssign(
                            annotation=ast.Name(id=nested_field.v_type, ctx=node.annotation.ctx),
                            simple=node.simple,
                            value=ast.Attribute(value=node.value, attr=nested_field.name, ctx=ctx),
                            target=ast.Attribute(value=node.target, attr=nested_field.name, ctx=ctx))
                        new_nodes.append(new_node)
                if len(new_nodes) > 0:
                    return new_nodes
        return node

    def visit_Assign(self, node):
        ret = []
        self.generic_visit(node)
        for field in node.targets:
            """
            No elimination for reference type
            """
            if (type(field) == ast.Attribute and field.attr.split("_")[-1] == "ref") \
            or (type(field) == ast.Name and field.id.split("_")[-1] == "ref"):
                return node
            if type(field) == ast.Attribute and type(self.node_path[-2]) == ClassNode:
                var_type = None
                class_name = self.node_path[-2].name
                for class_field in self.node_path[-2].declared_fields:
                    if class_field.name == field.attr:
                        var_type = class_field.v_type
                if node.value is not None and var_type in self.sdef_cls:
                    if type(node.value) == ast.Call and type(node.value.func) == ast.Name:
                        if node.value.func.id in self.sdef_cls:
                            func_body_gen = FunctionBodyGenerator(self.node, node.value.func.id)
                            func_body_gen.GetTransformedNodes(field,
                                                              "__init__",
                                                              node.value.args)
                            if len(func_body_gen.new_ast_nodes) != 0:
                                ret.extend(func_body_gen.new_ast_nodes)
                    else:
                        """
                        Added to handle assignments where the value is not a method call, for example:
                        -- other.force = new_force
                        """
                        field_class = self.root.GetClassNode(self.attribute_type(field))
                        if field_class is not None:
                            for nested_var in field_class.declared_fields:
                                ctx = node.value.ctx if hasattr(node.value, "ctx") else None
                                ret.append(ast.Assign(
                                    value=ast.Attribute(value=node.value, attr=nested_var.name, ctx=ctx),
                                    targets=[ast.Attribute(value=field, attr=nested_var.name, ctx=field.ctx)]
                                ))
        if len(ret) != 0:
            return ret
        return node

# TODO: turn this into nested compatible
# TODO: field synthesizing goes wrong -> join with inliner?
class FieldSynthesizer(DeviceCodeVisitor):
    def __init__(self, root: CallGraph, sdef_cls):
        super().__init__(root)
        self.sdef_cls = sdef_cls
    
    def visit_ClassDef(self, node):
        class_node = self.node_path[-1].GetClassNode(node.name)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        if class_node.is_device:
            self.node_path.append(class_node)
            # node.body = [self.visit(body) for body in node.body]
            node_body = []
            for body in node.body:
                rewritten = self.visit(body)
                if type(rewritten) == list:
                    for singular_body in rewritten:
                        node_body.append(singular_body)
                else:
                    node_body.append(rewritten)
            node.body= node_body
            self.node_path.pop()

    def visit_FunctionDef(self, node):
        func_node = self.node_path[-1].GetFunctionNode(node.name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)    
        if func_node.is_device:
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node_body = []
            if node.name == "__init__" and self.node_path[-2].has_random_state:
                node_body.append(ast.AnnAssign(
                    target=ast.Attribute(attr="random_state_", value=ast.Name("self")),
                    annotation=ast.Attribute(attr="RandomState", value=ast.Name(id="DeviceAllocator")),
                    simple=1,
                    value=ast.Constant(value=None, kind=None)
                ))
            for x in node.body:
                node_body.append(self.visit(x))
            node.body = node_body
            self.node_path.pop()           
        return node

    def visit_AnnAssign(self, node):
        # Replaces nested field access into a single field access joined by _
        self.generic_visit(node)
        # Remove type annotations from variable assignments
        if type(self.node_path[-1]) == FunctionNode and self.node_path[-1].name == "__init__":
            return node
        elif type(node.target) == ast.Attribute:
            return ast.Assign(targets=[node.target], value=node.value)
        node.simple = 1
        return node
    
    def visit_Attribute(self, node):
        ctx = None
        if type(node.ctx) == ast.Load:
            ctx = ast.Load()
        elif type(node.ctx) == ast.Store:
            ctx = ast.Store()
        if type(node.value) == ast.Name:
            if type(self.node_path[-1]) in [FunctionNode, ClassNode] \
            and self.node_path[-1].GetVariableType(node.value.id) in self.sdef_cls \
            and node.value.id.split("_")[-1] != "ref":
                return ast.Name(id=node.value.id + "_" + node.attr, ctx=ctx)
        elif type(node.value) == ast.Attribute and type(self.node_path[-2]) == ClassNode \
        and node.value.attr.split("_")[-1] != "ref":
            var_type = None
            if node.value.attr in self.node_path[-2].expanded_fields:
                for field in self.node_path[-2].declared_fields:
                    if field.name == node.value.attr:
                        var_type = field.v_type
            if var_type in self.sdef_cls:
                return ast.Attribute(attr=node.value.attr + "_" + node.attr, ctx=ctx, value=node.value.value)
        return node         

def transform(node, call_graph):
    """Replace all self-defined type fields with specific implementation"""
    ast.fix_missing_locations(Normalizer(call_graph).visit(node))
    gpcgv1 = GenPyCallGraphVisitor()
    gpcgv1.visit(node)
    if not gpcgv1.mark_device_data(node):
        print("No device data found")
        sys.exit(1)
    call_graph = gpcgv1.root
    scr = Searcher(call_graph)
    scr.visit(node)
    for cls in scr.dev_cls:
        if cls in scr.sdef_cls:
            scr.sdef_cls.remove(cls)
    ast.fix_missing_locations(Inliner(call_graph, node, scr.sdef_cls).visit(node))
    ast.fix_missing_locations(Eliminator(call_graph, node, scr.sdef_cls).visit(node))  
    ast.fix_missing_locations(FieldSynthesizer(call_graph, scr.sdef_cls).visit(node)) 

def compile(source_code, dir_path, file_name):
    """
    Compile python source_code into c++ source file and header file
        source_code:    codes written in python
    """
    # Set the global variable for file name
    FILE_NAME = file_name
    # Generate python ast
    py_ast = ast.parse(source_code)
    # Generate python call graph
    gpcgv = GenPyCallGraphVisitor()
    gpcgv.visit(py_ast)
    # Mark all device data on the call graph
    if not gpcgv.mark_device_data(py_ast):
        print("No device data found")
        sys.exit(1)
    # Transform nested-objects to basic fields
    transform(py_ast, gpcgv.root)
    new_py_code = astunparse.unparse(py_ast)
    new_py_ast = ast.parse(new_py_code)
    gpcgv1 = GenPyCallGraphVisitor()
    gpcgv1.visit(new_py_ast)
    # Mark all device data on the call graph
    if not gpcgv1.mark_device_data(new_py_ast):
        print("No device data found")
        sys.exit(1)
    # Generate cpp ast from python ast
    gcv = GenCppAstVisitor(gpcgv1.root)
    cpp_node = gcv.visit(new_py_ast)
    # Generate cpp(hpp) code from cpp ast
    ctx = build_cpp.BuildContext.create()
    # Expression needed for DynaSOAr API
    cpp_include_expr = '#include "{}.h"\n\n'.format(FILE_NAME)
    allocator_declaration = "AllocatorHandle<AllocatorT>* allocator_handle;\n" \
                            "__device__ AllocatorT* device_allocator;\n"
    precompile_expr = "#ifndef SANAJEH_DEVICE_CODE_H" \
                      "\n#define SANAJEH_DEVICE_CODE_H" \
                      "\n#define KNUMOBJECTS 64*64*64*64"
    hpp_include_expr = '\n\n#include <curand_kernel.h>\n#include "dynasoar.h"'
    init_cpp = ['\n\nextern "C" int AllocatorInitialize(){\n',
                INDENT +
                "allocator_handle = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);\n",
                INDENT +
                "AllocatorT* dev_ptr = allocator_handle->device_pointer();\n",
                INDENT +
                "cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);\n",
                gpcgv1.build_global_device_variables_init(),
                INDENT +
                "return 0;\n"
                "}"
                ]
    init_hpp = '\nextern "C" int AllocatorInitialize();\n'
    init_cdef = '\nint AllocatorInitialize();'
    unit_cpp = ['\n\nextern "C" int AllocatorUninitialize(){\n',
                gpcgv1.build_global_device_variables_unit(),
                INDENT +
                "return 0;\n"
                "}"
                ]
    unit_hpp = '\nextern "C" int AllocatorUninitialize();\n'
    unit_cdef = '\nint AllocatorUninitialize();'
    endif_expr = "\n#endif"

    # Source code
    cpp_code = cpp_include_expr \
               + allocator_declaration \
               + cpp_node.buildCpp(ctx) \
               + gpcgv1.build_do_all_cpp() \
               + gpcgv1.build_parallel_do_cpp() \
               + gpcgv1.build_parallel_new_cpp() \
               + "".join(init_cpp) \
               + "".join(unit_cpp)
    # Header code
    hpp_code = precompile_expr \
               + hpp_include_expr \
               + cpp_node.buildHpp(ctx) \
               + gpcgv1.build_do_all_hpp() \
               + gpcgv1.build_parallel_do_hpp() \
               + gpcgv1.build_parallel_new_hpp() \
               + init_hpp \
               + unit_hpp \
               + endif_expr

    # Codes for cffi cdef() function
    cdef_code = gpcgv1.build_parallel_do_cdef() \
                + gpcgv1.build_parallel_new_cdef() \
                + init_cdef + gpcgv1.build_do_all_cdef() \
                + unit_cdef
    """
    if not os.path.isdir(DIC_NAME):
      os.mkdir(DIC_NAME)
    with open(CPP_FILE_PATH, mode='w+') as cpp_file:
        cpp_file.write(cpp_code)
    with open(HPP_FILE_PATH, mode='w+') as hpp_file:
        hpp_file.write(hpp_code)
    with open(CDEF_FILE_PATH, mode='w+') as cdef_file:
        cdef_file.write(cdef_code)
    with open(PY_FILE_PATH, mode='w+') as py_file:
        py_file.write(new_py_code)
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    pth: str = dir_path + "/" + file_name
    with open(pth + ".cu", mode="w") as cpp_file:
        cpp_file.write(cpp_code)
    with open(pth + ".h", mode="w") as hpp_file:
        hpp_file.write(hpp_code)
    with open(pth + ".cdef", mode="w") as cdef_file:
        cdef_file.write(cdef_code)
    with open(pth + "_py.py", mode="w") as py_file:
        py_file.write(new_py_code)
    return cpp_code, hpp_code, cdef_code