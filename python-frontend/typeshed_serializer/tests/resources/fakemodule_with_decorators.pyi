import fakemodule_with_decorators_imported
from fakemodule_with_decorators_imported import another_imported_decorator 
from fakemodule_with_decorators_imported import alias_imported_decorator as alias_decorator

def a_method_decorator(a_meth):
    ...

def a_function_decorator(a_func):
    ...
    
@a_function_decorator
def a_common_function():
    ...

@fakemodule_with_decorators_imported.an_imported_decorator
def a_common_function2():
    ...
    
@another_imported_decorator
def a_common_function3():
    ...
    
@fakemodule_with_decorators_imported.another_imported_decorator
def a_common_function4():
    ...
    
@alias_decorator
def a_common_function5():
    ...
    
class CommonClass:
    @a_method_decorator
    def common_method(self):
        ...
        
    @fakemodule_with_decorators_imported.an_imported_decorator
    def common_method2(self):
        ...
        
    @another_imported_decorator
    def common_method3(self):
        ...

    @fakemodule_with_decorators_imported.another_imported_decorator
    def common_method4(self):
        ...
        
    @alias_decorator
    def common_method5(self):
        ...
    