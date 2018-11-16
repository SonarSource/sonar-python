/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.semantic;

import com.google.common.base.Functions;
import com.sonar.sslr.api.AstNode;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.sonar.python.PythonVisitor;
import org.sonar.python.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.api.PythonGrammar;

import static org.assertj.core.api.Assertions.assertThat;

public class SymbolTableBuilderVisitorTest {

  private SymbolTable symbolTable;
  private AstNode rootTree;
  private Map<String, AstNode> functionTreesByName = new HashMap<>();

  @Before
  public void init() {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(new File("src/test/resources/semantic/symbols.py"));
    SymbolTableBuilderVisitor symbolTableBuilderVisitor = new SymbolTableBuilderVisitor();
    symbolTableBuilderVisitor.scanFile(context);
    symbolTable = symbolTableBuilderVisitor.symbolTable();
    new TestVisitor().scanFile(context);
  }

  @Test
  public void non_scope_tree() throws Exception {
    assertThat(symbolTable.symbols(rootTree.getFirstDescendant(PythonGrammar.EXPRESSION_STMT))).isEmpty();
  }

  @Test
  public void module_variable() {
    assertThat(symbolTable.symbols(rootTree)).extracting(Symbol::name).containsOnly("a", "b", "t1");
    assertThat(symbolTable.symbols(rootTree)).extracting(Symbol::scopeTree).containsOnly(rootTree);
  }

  @Test
  public void local_variable() {
    AstNode functionTree = functionTreesByName.get("function_with_local");
    Set<Symbol> symbols = symbolsInFunction("function_with_local");
    Map<String, Symbol> symbolByName = symbols.stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("a", "t2");
    Symbol a = symbolByName.get("a");
    assertThat(a.scopeTree()).isEqualTo(functionTree);
    assertThat(a.writeUsages()).extracting(AstNode::getTokenLine).containsOnly(functionTree.getTokenLine() + 1);
    assertThat(a.readUsages()).extracting(AstNode::getTokenLine).containsOnly(
      functionTree.getTokenLine() + 2,
      functionTree.getTokenLine() + 3);
    Symbol t2 = symbolByName.get("t2");
    assertThat(t2.scopeTree()).isEqualTo(functionTree);
    assertThat(t2.writeUsages()).extracting(AstNode::getTokenLine).containsOnly(functionTree.getTokenLine() + 4);
    assertThat(t2.readUsages()).isEmpty();
  }

  @Test
  public void global_variable() {
    assertThat(symbolsInFunction("function_with_global")).extracting(Symbol::name).containsOnly("c", "t3");
  }

  @Test
  public void global_variable_reference() {
    Symbol a = lookup(rootTree, "a");
    AstNode nesting4 = functionTreesByName.get("nesting4");
    assertThat(a.writeUsages()).extracting(AstNode::getTokenLine).contains(1, nesting4.getTokenLine() + 2);
  }

  @Test
  public void nonlocal_variable() {
    assertThat(symbolsInFunction("function_with_nonlocal")).isEmpty();
  }

  @Test
  public void nonlobal_variable_reference() {
    Symbol a = symbolsInFunction("nesting2").iterator().next();
    AstNode function_with_nonlocal = functionTreesByName.get("function_with_nonlocal");
    assertThat(a.writeUsages()).extracting(AstNode::getTokenLine).contains(function_with_nonlocal.getTokenLine() + 2);
  }

  @Test
  public void compound_assignment() {
    assertThat(symbolsInFunction("compound_assignment")).extracting(Symbol::name).containsOnly("a");
  }

  @Test
  public void simple_parameter() {
    assertThat(symbolsInFunction("simple_parameter")).extracting(Symbol::name).containsOnly("a");
  }

  @Test
  public void list_parameter() {
    assertThat(symbolsInFunction("list_parameter")).extracting(Symbol::name).containsOnly("a", "b");
  }

  @Test
  public void dotted_name() {
    String functionName = "dotted_name";
    Set<Symbol> symbols = symbolsInFunction(functionName);
    assertThat(symbols).hasSize(1);
    Symbol a = symbols.iterator().next();
    assertThat(a.name()).isEqualTo("a");
    AstNode functionTree = functionTreesByName.get(functionName);
    assertThat(a.readUsages()).extracting(AstNode::getTokenLine).containsOnly(functionTree.getTokenLine() + 2);
  }

  @Test
  public void class_variable() {
    AstNode classC = rootTree.getFirstDescendant(PythonGrammar.CLASSDEF);
    assertThat(symbolTable.symbols(classC)).hasSize(3);
    Symbol classVariableA = lookup(classC, "a");
    assertThat(classVariableA.readUsages()).extracting(AstNode::getTokenLine).containsOnly(classC.getTokenLine() + 2);
    Symbol classVariableB = lookup(classC, "b");
    assertThat(classVariableB.readUsages()).extracting(AstNode::getTokenLine).containsOnly(classC.getTokenLine() + 3);
  }

  @Test
  public void global_variable_reference_in_class() {
    AstNode classC = rootTree.getFirstDescendant(PythonGrammar.CLASSDEF);
    Symbol a = lookup(rootTree, "a");
    assertThat(a.readUsages()).extracting(AstNode::getTokenLine).contains(classC.getTokenLine() + 1);
  }

  private Set<Symbol> symbolsInFunction(String functionName) {
    AstNode functionTree = functionTreesByName.get(functionName);
    return symbolTable.symbols(functionTree);
  }

  private Symbol lookup(AstNode scopeRootTree, String symbolName) {
    return symbolTable.symbols(scopeRootTree).stream()
      .filter(s -> s.name().equals(symbolName))
      .findFirst().get();
  }

  private class TestVisitor extends PythonVisitor {

    @Override
    public void visitFile(AstNode node) {
      rootTree = node;
      for (AstNode functionTree : node.getDescendants(PythonGrammar.FUNCDEF)) {
        String name = functionTree.getFirstChild(PythonGrammar.FUNCNAME).getTokenValue();
        functionTreesByName.put(name, functionTree);
      }
    }
  }

}
