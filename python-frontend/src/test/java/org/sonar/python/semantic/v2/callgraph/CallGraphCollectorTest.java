/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic.v2.callgraph;


import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

class CallGraphCollectorTest {
  @Test
  void collectCallGraph_singleFunctionCall() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      def caller():
          print('hello')
      
      def main():
          caller()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);
    
    assertThat(callGraph.getUsages("my_package.mod.caller")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.main");

    assertThat(callGraph.getUsages("my_package.mod.main")).isEmpty();
  }

  @Test
  void collectCallGraph_callBeforeDefinition() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      def main():
          caller()

      def caller():
          print('hello')
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);
    
    // wrong function order leads to unknown type of caller()
    assertThat(callGraph.getUsages("my_package.mod.caller")).isEmpty();
  }

  @Test
  void collectCallGraph_multipleFunctionCalls() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      def target():
          pass
      
      def caller1():
          target()
      
      def caller2():
          target()
      
      def caller3():
          caller1()
          caller2()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);
    
    assertThat(callGraph.getUsages("my_package.mod.target")).extracting(CallGraphNode::fqn)
      .containsExactlyInAnyOrder("my_package.mod.caller1", "my_package.mod.caller2");
    
    assertThat(callGraph.getUsages("my_package.mod.caller1")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.caller3");
    
    assertThat(callGraph.getUsages("my_package.mod.caller2")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.caller3");

    assertThat(callGraph.getUsages("my_package.mod.caller3")).isEmpty();
  }

  @Test
  void collectCallGraph_nestedFunctionCalls() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      def outer():
          def inner():
              pass
          inner()
      
      def other():
          outer()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);
    
    assertThat(callGraph.getUsages("my_package.mod.outer.inner")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.outer");
    
    assertThat(callGraph.getUsages("my_package.mod.outer")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.other");
  }

  @Test
  void collectCallGraph_classConstructor() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      def foo():
        pass

      class MyClass:
        def __init__(self):
          foo()
      
      def bar():
        obj = MyClass()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);

    assertThat(callGraph.getUsages("my_package.mod.foo")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.MyClass.__init__");

    // Class constructor usages aren't tracked in the call graph
    assertThat(callGraph.getUsages("my_package.mod.MyClass.__init__")).isEmpty();
  }

  @Test
  void collectCallGraph_methodCalls() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      class MyClass:
        def method1(self):
            return 42
      
      def function():
        obj = MyClass()
        obj.method1()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);

    assertThat(callGraph.getUsages("my_package.mod.MyClass.__init__")).isEmpty();

    assertThat(callGraph.getUsages("my_package.mod.MyClass.method1")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.function");
  }

  @Test
  void collectCallGraph_selfMethodCall() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      class MyClass:
        def method1(self):
            return 42

        def method2(self):
            aaaa.method1()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);

    // self is not inferred
    assertThat(callGraph.getUsages("my_package.mod.MyClass.method1")).isEmpty();
  }

  @Test
  void collectCallGraph_builtinFunctionCalls() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      import os
      from math import sqrt
      
      def my_function():
          print('hello')
          len([1, 2, 3])
          
          os.path.join('a', 'b')
          sqrt(16)
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);
    
    assertThat(callGraph.getUsages("print")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.my_function");
    
    assertThat(callGraph.getUsages("len")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.my_function");
    
    assertThat(callGraph.getUsages("posixpath.join")).extracting(CallGraphNode::fqn)
        .containsExactly("my_package.mod.my_function");

    assertThat(callGraph.getUsages("math.sqrt")).extracting(CallGraphNode::fqn)
        .containsExactly("my_package.mod.my_function");
  }


  @Test
  void collectCallGraph_recursiveFunction() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      def factorial(n):
          if n <= 1:
              return 1
          return n * factorial(n - 1)
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);
    
    assertThat(callGraph.getUsages("my_package.mod.factorial")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.factorial");
  }

  @Test
  void collectCallGraph_lambdaFunction() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      def helper():
          pass
      
      def main():
          func = lambda: helper()
          func()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);
    
    assertThat(callGraph.getUsages("my_package.mod.helper")).isEmpty(); // lambdas don't have an FQN at the moment
    assertThat(callGraph.getUsages("my_package.mod.main")).isEmpty();
    assertThat(callGraph.getUsages("my_package.mod.func")).isEmpty(); 
  }

  @Test
  void collectCallGraph_unknownFunction() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      from some_module import a_function
      def my_function():
          unknown_function()
          a_function()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);

    assertThat(callGraph.getUsages("some_module.a_function"))
      .extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.my_function");
    
    // Unknown symbols don't have an FQN; thus they don't show up in the graph
    assertThat(callGraph.getUsages("unknown_function")).isEmpty();
  }

  @Test
  void collectCallGraph_multipleFiles() {
    var typeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    var otherFile = PythonTestUtils.pythonFile("other");
    var modFile = PythonTestUtils.pythonFile("mod");

    TypesTestUtils.parseAndInferTypes(typeTable, otherFile, "def a_function(): pass");
    FileInput fileInput = TypesTestUtils.parseAndInferTypes(typeTable, modFile, """
      from other import a_function
      def my_function():
        a_function()
      """);

    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);

    assertThat(callGraph.getUsages("other.a_function")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.my_function");
  }

  @Test
  void collectCallGraph_callModule() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      import math
      def my_function():
        math()
      """);
    CallGraph callGraph = CallGraphCollector.collectCallGraph(fileInput);

    assertThat(callGraph.getUsages("math")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.mod.my_function");
  }
}
