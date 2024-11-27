/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic;

import com.google.common.base.Functions;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.tree.TreeUtils;

import static org.assertj.core.api.Assertions.assertThat;

class SymbolTableBuilderTest {
  private static Map<String, FunctionDef> functionTreesByName = new HashMap<>();
  private static FileInput fileInput;


  private Map<String, Symbol> getSymbolByName(FunctionDef functionTree) {
    return functionTree.localVariables().stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
  }

  @BeforeAll
  static void init() {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(new File("src/test/resources/semantic/symbols2.py"));
    fileInput = context.rootTree();
    fileInput.accept(new TestVisitor());
  }

  @Test
  void global_variable() {
    Set<Symbol> moduleSymbols = fileInput.globalVariables();
    List<String> topLevelFunctions = Arrays.asList("function_with_local", "function_with_free_variable", "function_with_rebound_variable",
      "ref_in_interpolated", "print_var", "function_with_global_var", "func_wrapping_class", "function_with_unused_import",
      "function_with_nonlocal_var", "symbols_in_comp", "scope_of_comprehension", "for_comp_with_no_name_var",
      "function_with_loops", "simple_parameter", "comprehension_reusing_name", "tuple_assignment", "function_with_comprehension",
      "binding_usages", "func_with_star_param", "multiple_assignment", "function_with_nested_nonlocal_var", "func_with_tuple_param",
      "function_with_lambdas", "var_with_usages_in_decorator", "fn_inside_comprehension_same_name", "with_instance", "exception_instance", "unpacking",
      "using_builtin_symbol", "keyword_usage", "comprehension_vars", "parameter_default_value", "assignment_expression", "assignment_expression_in_generator",
      "assignment_expression_in_list_comprehension", "assignment_expression_in_set_comprehension", "assignment_expression_in_dict_comprehension",
      "importing_stdlib", "importing_submodule", "importing_submodule_as", "importing_submodule_after_parent", "importing_submodule_after_parent_nested",
      "importing_parent_after_submodule", "importing_parent_after_submodule_2", "importing_submodule_twice", "importing_unknown_submodule", "type_params", "type_alias");

    List<String> globalSymbols = new ArrayList<>(topLevelFunctions);
    globalSymbols.addAll(Arrays.asList("a", "global_x", "global_var"));

    assertThat(moduleSymbols)
      .filteredOn(symbol -> topLevelFunctions.contains(symbol.name()))
      .extracting(Symbol::kind).containsOnly(Symbol.Kind.FUNCTION);

    assertThat(moduleSymbols).extracting(Symbol::name).containsExactlyInAnyOrder(globalSymbols.toArray(new String[]{}));
    moduleSymbols.stream().filter(s -> s.name().equals("global_var")).findFirst().ifPresent(s -> {
      assertThat(s.usages()).hasSize(3);
    });
  }

  @Test
  void local_variable() {
    FunctionDef functionTree = functionTreesByName.get("function_with_local");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName.keySet()).containsOnly("a", "t2");
    Symbol a = symbolByName.get("a");
    Name aTree = PythonTestUtils.getFirstChild(functionTree, t -> t.is(Tree.Kind.NAME) && ((Name) t).name().equals("a"));
    assertThat(aTree.symbol()).isEqualTo(a);
    int functionStartLine = functionTree.firstToken().line();
    assertThat(a.usages()).extracting(usage -> usage.tree().firstToken().line()).containsOnly(
      functionStartLine + 1, functionStartLine + 2, functionStartLine + 3, functionStartLine + 4);
    assertThat(a.usages()).extracting(Usage::kind).containsOnly(
      Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.COMPOUND_ASSIGNMENT_LHS, Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.OTHER);
    Symbol t2 = symbolByName.get("t2");
    assertThat(t2.usages()).extracting(usage -> usage.tree().firstToken().line()).containsOnly(
      functionStartLine + 5);
    assertThat(t2.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.ASSIGNMENT_LHS);
  }

  @Test
  void free_variable() {
    FunctionDef functionTree = functionTreesByName.get("function_with_free_variable");
    assertThat(functionTree.localVariables()).isEmpty();
  }

  @Test
  void rebound_variable() {
    FunctionDef functionTree = functionTreesByName.get("function_with_rebound_variable");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName.keySet()).containsOnly("global_x");
  }

  @Test
  void simple_parameter() {
    FunctionDef functionTree = functionTreesByName.get("simple_parameter");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName.keySet()).containsOnly("a");
  }

  @Test
  void multiple_assignment() {
    FunctionDef functionTree = functionTreesByName.get("multiple_assignment");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName.keySet()).containsOnly("x", "y");
    int functionStartLine = functionTree.firstToken().line();
    Symbol x = symbolByName.get("x");
    assertThat(x.usages()).extracting(usage -> usage.tree().firstToken().line()).containsOnly(functionStartLine + 1);
    assertThat(x.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.ASSIGNMENT_LHS);
    Symbol y = symbolByName.get("y");
    assertThat(y.usages()).extracting(usage -> usage.tree().firstToken().line()).containsOnly(functionStartLine + 1);
    assertThat(y.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.ASSIGNMENT_LHS);
  }

  @Test
  void tuple_assignment() {
    FunctionDef functionTree = functionTreesByName.get("tuple_assignment");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName.keySet()).containsOnly("x", "y");
    int functionStartLine = functionTree.firstToken().line();
    Symbol x = symbolByName.get("x");
    assertThat(x.usages()).extracting(usage -> usage.tree().firstToken().line()).containsOnly(functionStartLine + 1);
    assertThat(x.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.ASSIGNMENT_LHS);
    Symbol y = symbolByName.get("y");
    assertThat(y.usages()).extracting(usage -> usage.tree().firstToken().line()).containsOnly(functionStartLine + 1);
    assertThat(y.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.ASSIGNMENT_LHS);
  }

  @Test
  void function_with_global_var() {
    FunctionDef functionTree = functionTreesByName.get("function_with_global_var");
    assertThat(functionTree.localVariables()).isEmpty();
  }

  @Test
  void function_with_nonlocal_var() {
    FunctionDef functionTree = functionTreesByName.get("function_with_nonlocal_var");
    assertThat(functionTree.localVariables()).isEmpty();
  }

  @Test
  void function_with_nested_nonlocal_var() {
    FunctionDef functionTree = functionTreesByName.get("function_with_nested_nonlocal_var");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName.keySet()).containsExactly("x", "innerFn");
    Symbol x = symbolByName.get("x");
    int functionStartLine = functionTree.firstToken().line();
    assertThat(x.usages()).extracting(usage -> usage.tree().firstToken().line()).containsOnly(functionStartLine + 1, functionStartLine + 3, functionStartLine + 4);
    FunctionDef innerFunctionTree = functionTreesByName.get("innerFn");
    assertThat(innerFunctionTree.localVariables()).isEmpty();
  }

  @Test
  void lambdas() {
    FunctionDef functionTree = functionTreesByName.get("function_with_lambdas");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);

    assertThat(symbolByName.keySet()).containsOnly("x", "y");
    Symbol x = symbolByName.get("x");
    assertThat(x.usages()).hasSize(1);

    Symbol y = symbolByName.get("y");
    assertThat(y.usages()).hasSize(2);

    List<LambdaExpression> lambdas = PythonTestUtils.getAllDescendant(functionTree, t -> t.is(Tree.Kind.LAMBDA));
    LambdaExpression firstLambdaFunction = lambdas.get(0);
    symbolByName = firstLambdaFunction.localVariables().stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("x");
    Symbol innerX = symbolByName.get("x");
    assertThat(innerX.usages()).hasSize(3);

    LambdaExpression secondLambdaFunction = lambdas.get(1);
    symbolByName = secondLambdaFunction.localVariables().stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("z");
  }

  @Test
  void for_stmt() {
    FunctionDef functionTree = functionTreesByName.get("function_with_loops");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);

    assertThat(symbolByName.keySet()).containsOnly("x", "y", "x1", "y1");
    Symbol x = symbolByName.get("x");
    assertThat(x.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.LOOP_DECLARATION);
    Symbol y = symbolByName.get("y");
    assertThat(y.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.LOOP_DECLARATION);
    Symbol x1 = symbolByName.get("x1");
    assertThat(x1.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.LOOP_DECLARATION);
    Symbol y1 = symbolByName.get("y1");
    assertThat(y1.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.LOOP_DECLARATION);
  }

  @Test
  void comprehension() {
    FunctionDef functionTree = functionTreesByName.get("function_with_comprehension");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).isEmpty();
    List<Name> names = getNameFromExpression(((ComprehensionExpression) ((ExpressionStatement) functionTree.body().statements().get(0)).expressions().get(0)).comprehensionFor().loopExpression());
    assertThat(names).hasSize(1).extracting(Name::name).containsOnly("a");
    Symbol a = names.get(0).symbol();
    assertThat(a.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.COMP_DECLARATION);
  }

  @Test
  void func_wrapping_class() {
    FunctionDef functionTree = functionTreesByName.get("func_wrapping_class");
    assertThat(functionTree.localVariables()).extracting(Symbol::name).containsExactly("A");
  }

  @Test
  void var_with_usages_in_decorator() {
    FunctionDef functionTree = functionTreesByName.get("var_with_usages_in_decorator");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);

    assertThat(symbolByName.keySet()).containsOnly("x", "y", "z", "foo");
    Symbol x = symbolByName.get("x");
    assertThat(x.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.OTHER);
    Symbol y = symbolByName.get("y");
    assertThat(y.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.OTHER);
    Symbol z= symbolByName.get("z");
    assertThat(z.usages()).extracting(Usage::kind).containsOnly(Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.OTHER);
  }

  @Test
  void function_with_unused_import() {
    FunctionDef functionTree = functionTreesByName.get("function_with_unused_import");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);

    assertThat(symbolByName.keySet()).containsOnly("mod1", "aliased_mod2", "x", "z");
    assertThat(symbolByName.get("mod1").usages()).extracting(Usage::kind).containsOnly(Usage.Kind.IMPORT);
    assertThat(symbolByName.get("aliased_mod2").usages()).extracting(Usage::kind).containsOnly(Usage.Kind.IMPORT);

    assertThat(symbolByName.get("x").usages()).extracting(Usage::kind).containsOnly(Usage.Kind.IMPORT);
    assertThat(symbolByName.get("z").usages()).extracting(Usage::kind).containsOnly(Usage.Kind.IMPORT);
  }

  @Test
  void importing_stdlib() {
    FunctionDef functionDef = functionTreesByName.get("importing_stdlib");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("os");
    assertThat(symbolByName.get("os").usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.OTHER);

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(1)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol).isNotNull();
    assertThat(qualifiedExpressionSymbol.kind()).isEqualTo(Symbol.Kind.AMBIGUOUS);
    Symbol symbol = ((AmbiguousSymbolImpl) qualifiedExpressionSymbol).alternatives().iterator().next();
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.FUNCTION);
  }

  @Test
  void importing_submodule() {
    FunctionDef functionDef = functionTreesByName.get("importing_submodule");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("werkzeug");
    SymbolImpl werkzeug = (SymbolImpl) symbolByName.get("werkzeug");
    assertThat(werkzeug.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.OTHER);
    Map<String, Symbol> childrenSymbolByName = werkzeug.getChildrenSymbolByName();
    assertThat(childrenSymbolByName.values()).extracting(Symbol::name).containsExactly("datastructures");
    SymbolImpl datastructures = (SymbolImpl) childrenSymbolByName.get("datastructures");
    assertThat(datastructures.getChildrenSymbolByName().values()).extracting(Symbol::name).contains("Headers");

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(1)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol.is(Symbol.Kind.CLASS)).isTrue();
    assertThat(qualifiedExpressionSymbol.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.headers.Headers");
  }

  @Test
  void importing_submodule_as() {
    FunctionDef functionDef = functionTreesByName.get("importing_submodule_as");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("wd");
    SymbolImpl werkzeug = (SymbolImpl) symbolByName.get("wd");
    assertThat(werkzeug.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.OTHER);
    Map<String, Symbol> childrenSymbolByName = werkzeug.getChildrenSymbolByName();
    Collection<Symbol> values = childrenSymbolByName.values();
    assertThat(values).isNotEmpty();
    assertThat(values).extracting(Symbol::name).contains("Headers");

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(1)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol.is(Symbol.Kind.CLASS)).isTrue();
    assertThat(qualifiedExpressionSymbol.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.headers.Headers");
  }

  @Test
  void importing_submodule_after_parent() {
    FunctionDef functionDef = functionTreesByName.get("importing_submodule_after_parent");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("werkzeug");
    SymbolImpl werkzeug = (SymbolImpl) symbolByName.get("werkzeug");
    assertThat(werkzeug.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.IMPORT, Usage.Kind.OTHER);
    Map<String, Symbol> childrenSymbolByName = werkzeug.getChildrenSymbolByName();
    assertThat(childrenSymbolByName.values()).extracting(Symbol::name).contains("datastructures");
    SymbolImpl datastructures = (SymbolImpl) childrenSymbolByName.get("datastructures");
    assertThat(datastructures.getChildrenSymbolByName().values()).extracting(Symbol::name).contains("Headers");

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(2)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol.is(Symbol.Kind.CLASS)).isTrue();
    assertThat(qualifiedExpressionSymbol.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.headers.Headers");
  }

  @Test
  void importing_submodule_after_parent_nested() {
    FunctionDef functionDef = functionTreesByName.get("importing_submodule_after_parent_nested");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("werkzeug");
    SymbolImpl werkzeug = (SymbolImpl) symbolByName.get("werkzeug");
    assertThat(werkzeug.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.IMPORT, Usage.Kind.IMPORT, Usage.Kind.OTHER);
    Map<String, Symbol> childrenSymbolByName = werkzeug.getChildrenSymbolByName();
    assertThat(childrenSymbolByName.values()).extracting(Symbol::name).contains("datastructures");
    SymbolImpl datastructures = (SymbolImpl) childrenSymbolByName.get("datastructures");
    assertThat(datastructures.getChildrenSymbolByName().values()).extracting(Symbol::name).contains("Headers", "csp");

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(3)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol.is(Symbol.Kind.CLASS)).isTrue();
    assertThat(qualifiedExpressionSymbol.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.csp.ContentSecurityPolicy");
  }

  @Test
  void importing_parent_after_submodule() {
    FunctionDef functionDef = functionTreesByName.get("importing_parent_after_submodule");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("werkzeug");
    SymbolImpl werkzeug = (SymbolImpl) symbolByName.get("werkzeug");
    assertThat(werkzeug.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.IMPORT, Usage.Kind.OTHER);
    Map<String, Symbol> childrenSymbolByName = werkzeug.getChildrenSymbolByName();
    assertThat(childrenSymbolByName.values()).extracting(Symbol::name).contains("datastructures");
    SymbolImpl datastructures = (SymbolImpl) childrenSymbolByName.get("datastructures");
    assertThat(datastructures.getChildrenSymbolByName().values()).extracting(Symbol::name).contains("csp");
    assertThat(datastructures.getChildrenSymbolByName().values()).extracting(Symbol::name).doesNotContain("Headers");

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(2)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol.is(Symbol.Kind.CLASS)).isTrue();
    assertThat(qualifiedExpressionSymbol.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.csp.ContentSecurityPolicy");
  }

  @Test
  void importing_parent_after_submodule_2() {
    FunctionDef functionDef = functionTreesByName.get("importing_parent_after_submodule_2");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("werkzeug");
    SymbolImpl werkzeug = (SymbolImpl) symbolByName.get("werkzeug");
    assertThat(werkzeug.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.IMPORT, Usage.Kind.IMPORT, Usage.Kind.OTHER, Usage.Kind.OTHER);
    Map<String, Symbol> childrenSymbolByName = werkzeug.getChildrenSymbolByName();
    assertThat(childrenSymbolByName.values()).extracting(Symbol::name).contains("datastructures", "Response");
    SymbolImpl datastructures = (SymbolImpl) childrenSymbolByName.get("datastructures");
    assertThat(datastructures.getChildrenSymbolByName().values()).extracting(Symbol::name).contains("Headers", "csp");

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(3)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol.is(Symbol.Kind.CLASS)).isTrue();
    assertThat(qualifiedExpressionSymbol.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.csp.ContentSecurityPolicy");

    CallExpression callExpression2 = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(4)).expressions().get(0);
    Symbol qualifiedExpressionSymbol2 = callExpression2.calleeSymbol();
    assertThat(qualifiedExpressionSymbol2.is(Symbol.Kind.CLASS)).isTrue();
    assertThat(qualifiedExpressionSymbol2.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.headers.Headers");
  }

  @Test
  void importing_submodule_twice() {
    FunctionDef functionDef = functionTreesByName.get("importing_submodule_twice");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("werkzeug");
    SymbolImpl werkzeug = (SymbolImpl) symbolByName.get("werkzeug");
    assertThat(werkzeug.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.IMPORT, Usage.Kind.OTHER);
    Map<String, Symbol> childrenSymbolByName = werkzeug.getChildrenSymbolByName();
    assertThat(childrenSymbolByName.values()).extracting(Symbol::name).containsExactly("datastructures");
    SymbolImpl datastructures = (SymbolImpl) childrenSymbolByName.get("datastructures");
    assertThat(datastructures.getChildrenSymbolByName().values()).extracting(Symbol::name).contains("Headers");

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(2)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol.is(Symbol.Kind.CLASS)).isTrue();
    assertThat(qualifiedExpressionSymbol.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.headers.Headers");
  }

  @Test
  void importing_unknown_submodule() {
    FunctionDef functionDef = functionTreesByName.get("importing_unknown_submodule");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);

    assertThat(symbolByName).containsOnlyKeys("werkzeug");
    SymbolImpl werkzeug = (SymbolImpl) symbolByName.get("werkzeug");
    assertThat(werkzeug.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT, Usage.Kind.IMPORT, Usage.Kind.OTHER);
    Map<String, Symbol> childrenSymbolByName = werkzeug.getChildrenSymbolByName();
    assertThat(childrenSymbolByName.values()).extracting(Symbol::name).contains("datastructures");
    SymbolImpl datastructures = (SymbolImpl) childrenSymbolByName.get("datastructures");
    assertThat(datastructures.getChildrenSymbolByName().values()).extracting(Symbol::name).containsExactly("Headers");

    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionDef.body().statements().get(2)).expressions().get(0);
    Symbol qualifiedExpressionSymbol = callExpression.calleeSymbol();
    assertThat(qualifiedExpressionSymbol.is(Symbol.Kind.OTHER)).isTrue();
    assertThat(qualifiedExpressionSymbol.fullyQualifiedName()).isEqualTo("werkzeug.datastructures.Headers");
  }

  @Test
  void function_with_tuple_param() {
    FunctionDef functionTree = functionTreesByName.get("func_with_tuple_param");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(4);
  }

  @Test
  void function_with_star_param() {
    FunctionDef functionTree = functionTreesByName.get("func_with_star_param");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(2);

    functionTree = functionTreesByName.get("method_with_star_param");
    symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(1);
  }

  @Test
  void print_local_var() {
    FunctionDef functionTree = functionTreesByName.get("print_var");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(1).containsOnlyKeys("print");
  }

  @Test
  void tuples_in_comp() {
    FunctionDef functionTree = functionTreesByName.get("symbols_in_comp");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).isEmpty();
    List<Name> names = getNameFromExpression(((ComprehensionExpression) ((ExpressionStatement) functionTree.body().statements().get(0)).expressions().get(0)).comprehensionFor().loopExpression());
    assertThat(names).hasSize(3).extracting(Name::name).containsOnly("x", "y", "z");
    for (Symbol symbol : symbolByName.values()) {
      assertThat(symbol.usages()).hasSize(2);
    }
  }

  @Test
  void comprehension_scope() {
    FunctionDef functionTree = functionTreesByName.get("scope_of_comprehension");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).containsOnlyKeys("x");
    assertThat(symbolByName.get("x").usages()).extracting(u -> u.tree().firstToken().line()).containsExactly(108, 110).doesNotContain(109);
    Name name = (Name) ((ComprehensionExpression) ((ExpressionStatement) functionTree.body().statements().get(0)).expressions().get(0)).comprehensionFor().loopExpression();
    assertThat(name.symbol().usages()).extracting(u -> u.tree().firstToken().line()).doesNotContain(108, 110).containsExactly(109, 109);
  }

  @Test
  void comprehension_shadowing_names() {
    FunctionDef functionTree = functionTreesByName.get("comprehension_reusing_name");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(1);
    assertThat(symbolByName.get("a").usages()).hasSize(2);
    List<Name> names = getNameFromExpression(((DictCompExpression) ((ExpressionStatement) functionTree.body().statements().get(0)).expressions().get(0)).comprehensionFor().loopExpression());
    assertThat(names).hasSize(1);
    assertThat(names.get(0).symbol().usages()).hasSize(2);
  }

  @Test
  void interpolated_string() {
    FunctionDef functionTree = functionTreesByName.get("ref_in_interpolated");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(1);
    assertThat(symbolByName.get("p1").usages()).hasSize(2);
  }

  @Test
  void fn_inside_comprehension_same_name() {
    FunctionDef functionTree = functionTreesByName.get("fn_inside_comprehension_same_name");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(1);
    assertThat(symbolByName.get("fn").usages()).extracting(Usage::kind).containsExactly(Usage.Kind.FUNC_DECLARATION);
  }

  @Test
  void exception_instance() {
    FunctionDef functionTree = functionTreesByName.get("exception_instance");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(6);
    assertThat(symbolByName.get("e1").usages()).extracting(Usage::kind).containsExactly(Usage.Kind.EXCEPTION_INSTANCE);
  }

  @Test
  void with_instance() {
    FunctionDef functionTree = functionTreesByName.get("with_instance");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(3);
    assertThat(symbolByName.get("file1").usages()).extracting(Usage::kind).containsExactly(Usage.Kind.WITH_INSTANCE);
  }

  @Test
  void unpacking() {
    FunctionDef functionTree = functionTreesByName.get("unpacking");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(1);
    assertThat(symbolByName.get("foo").usages()).extracting(Usage::kind).containsExactly(Usage.Kind.ASSIGNMENT_LHS);
  }

  @Test
  void using_builtin_symbol() {
    FunctionDef functionTree = functionTreesByName.get("using_builtin_symbol");
    CallExpression callExpression = (CallExpression) ((ExpressionStatement) functionTree.body().statements().get(0)).expressions().get(0);
    Name print = (Name) callExpression.callee();
    assertThat(print.symbol()).isNotNull();
    assertThat(print.symbol().name()).isEqualTo("print");
    assertThat(print.symbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.OTHER);
    assertThat(print.symbol().fullyQualifiedName()).isEqualTo("print");
  }

  @Test
  void keyword_usage() {
    FunctionDef functionTree = functionTreesByName.get("keyword_usage");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(1);
    Symbol x = symbolByName.get("x");
    assertThat(x.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.ASSIGNMENT_LHS);
  }

  @Test
  void parameter_default_value() {
    FunctionDef functionTree = functionTreesByName.get("parameter_default_value");
    Map<String, Symbol> symbolByName = getSymbolByName(functionTree);
    assertThat(symbolByName).hasSize(2);
    Symbol foo = symbolByName.get("foo");
    assertThat(foo.usages()).hasSize(2);
    Usage assignmentUsage = foo.usages().get(0);
    assertThat(assignmentUsage.kind()).isEqualTo(Usage.Kind.ASSIGNMENT_LHS);
    Usage parameterUsage = foo.usages().get(1);
    assertThat(parameterUsage.kind()).isEqualTo(Usage.Kind.OTHER);
    assertThat(TreeUtils.firstAncestorOfKind(parameterUsage.tree(), Tree.Kind.PARAMETER)).isNotNull();

    Symbol func = symbolByName.get("func");
    assertThat(func.usages()).hasSize(1);
    FunctionDef functionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(func.usages().get(0).tree(), Tree.Kind.FUNCDEF);
    assertThat(functionDef).isNotNull();
    assertThat(functionDef.localVariables()).hasSize(2);

    Symbol foo2 = functionDef.localVariables().stream().filter(s -> s.name().equals("foo")).findFirst().get();
    assertThat(foo2.name()).isEqualTo("foo");
    assertThat(foo2.usages()).hasSize(1);
    assertThat(foo2.usages().get(0).kind()).isEqualTo(Usage.Kind.ASSIGNMENT_LHS);
  }

  @Test
  void comprehension_vars() {
    FunctionDef functionTree = functionTreesByName.get("comprehension_vars");
    ComprehensionExpression comprehensionExpression = ((ComprehensionExpression) ((ExpressionStatement) functionTree.body().statements().get(0)).expressions().get(0));
    assertThat(comprehensionExpression.localVariables()).hasSize(1);
    Map<String, Symbol> symbolByName = comprehensionExpression.localVariables().stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    Symbol a = symbolByName.get("a");
    assertThat(a.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.COMP_DECLARATION);
  }

  @Test
  void assignment_expression() {
    FunctionDef functionDef = functionTreesByName.get("assignment_expression");
    assertThat(functionDef.localVariables()).hasSize(1);
    Symbol b = functionDef.localVariables().iterator().next();
    assertThat(b.name()).isEqualTo("b");
    assertThat(b.fullyQualifiedName()).isNull();
    assertThat(b.usages()).hasSize(2);
    assertThat(b.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.OTHER);
  }

  @ValueSource(strings = {"assignment_expression_in_generator", "assignment_expression_in_list_comprehension", "assignment_expression_in_set_comprehension", "assignment_expression_in_dict_comprehension"})
  @ParameterizedTest
  void assignment_expression_in_comprehension(String functionName) {
    FunctionDef functionDef = functionTreesByName.get(functionName);
    assertThat(functionDef.localVariables()).hasSize(1);
    Symbol comment = functionDef.localVariables().iterator().next();
    assertThat(comment.name()).isEqualTo("last");
    assertThat(comment.fullyQualifiedName()).isNull();
    assertThat(comment.usages()).hasSize(2);
    assertThat(comment.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.OTHER);
  }

  @Test
  void type_params() {
    FunctionDef functionDef = functionTreesByName.get("type_params");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);
    assertThat(symbolByName).hasSize(2).containsKey("T");
    assertThat(symbolByName.get("T").usages()).hasSize(2);
    assertThat(symbolByName.get("T").usages().get(0).kind()).isEqualTo(Usage.Kind.TYPE_PARAM_DECLARATION);
  }

  @Test
  void type_alias_declaration() {
    FunctionDef functionDef = functionTreesByName.get("type_alias");
    Map<String, Symbol> symbolByName = getSymbolByName(functionDef);
    assertThat(symbolByName).hasSize(1).containsKey("M");
    assertThat(symbolByName.get("M").usages()).hasSize(2);
    assertThat(symbolByName.get("M").usages().get(0).kind()).isEqualTo(Usage.Kind.TYPE_ALIAS_DECLARATION);
  }

  private static class TestVisitor extends BaseTreeVisitor {
    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      functionTreesByName.put(pyFunctionDefTree.name().name(), pyFunctionDefTree);
      super.visitFunctionDef(pyFunctionDefTree);
    }
  }

  private static List<Name> getNameFromExpression(Tree tree) {
    List<Name> res = new ArrayList<>();
    if (tree.is(Tree.Kind.NAME)) {
      res.add(((Name) tree));
    } else if (tree.is(Tree.Kind.TUPLE)) {
      ((Tuple) tree).elements().forEach(t -> res.addAll(getNameFromExpression(t)));
    }
    return res;
  }
}
