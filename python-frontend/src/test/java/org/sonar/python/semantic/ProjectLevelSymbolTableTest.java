/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.semantic.ProjectLevelSymbolTable.empty;
import static org.sonar.python.semantic.ProjectLevelSymbolTable.from;

public class ProjectLevelSymbolTableTest {

  private Map<String, Symbol> getSymbolByName(FileInput fileInput) {
    return fileInput.globalVariables().stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
  }

  @Test
  public void wildcard_import() {
    SymbolImpl exportedA = new SymbolImpl("a", "mod.a");
    SymbolImpl exportedB = new SymbolImpl("b", "mod.b");
    SymbolImpl exportedC = new ClassSymbolImpl("C", "mod.C");
    List<Symbol> modSymbols = Arrays.asList(exportedA, exportedB, exportedC);
    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("mod", new HashSet<>(modSymbols));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "from mod import *",
      "print(a)"
    );
    assertThat(tree.globalVariables()).extracting(Symbol::name).containsExactlyInAnyOrder("a", "b", "C");
    Symbol a = getSymbolByName(tree).get("a");
    assertThat(exportedA.usages()).isEmpty();
    assertThat(a).isNotEqualTo(exportedA);
    assertThat(a.fullyQualifiedName()).isEqualTo("mod.a");
    assertThat(a.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.OTHER, Usage.Kind.IMPORT);

    Symbol b = getSymbolByName(tree).get("b");
    assertThat(exportedB.usages()).isEmpty();
    assertThat(b).isNotEqualTo(exportedB);
    assertThat(b.fullyQualifiedName()).isEqualTo("mod.b");
    assertThat(b.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT);

    Symbol c = getSymbolByName(tree).get("C");
    assertThat(exportedC.usages()).isEmpty();
    assertThat(c).isNotEqualTo(exportedC);
    assertThat(c.fullyQualifiedName()).isEqualTo("mod.C");
    assertThat(c.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT);
  }

  @Test
  public void unresolved_wildcard_import() {
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), empty()),
      "from external import *",
      "print(a)"
    );
    ImportFrom importFrom = ((ImportFrom) PythonTestUtils.getAllDescendant(tree, t -> t.is(Tree.Kind.IMPORT_FROM)).get(0));
    assertThat(importFrom.hasUnresolvedWildcardImport()).isTrue();
  }

  @Test
  public void function_symbol() {
    FunctionDef functionDef = (FunctionDef) parse("def fn(p1, p2): pass").statements().statements().get(0);
    FunctionSymbolImpl fnSymbol = new FunctionSymbolImpl(functionDef, "mod.fn", pythonFile("mod.py"));
    List<Symbol> modSymbols = Collections.singletonList(fnSymbol);
    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("mod", new HashSet<>(modSymbols));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "from mod import fn",
      "fn(1, 2)"
    );
    CallExpression callExpression = PythonTestUtils.getFirstChild(tree, t -> t.is(Tree.Kind.CALL_EXPR));
    Symbol importedFnSymbol = callExpression.calleeSymbol();
    assertThat(importedFnSymbol).isNotEqualTo(fnSymbol);
    assertThat(importedFnSymbol.kind()).isEqualTo(Symbol.Kind.FUNCTION);
    assertThat(fnSymbol.usages()).isEmpty();
    assertThat(importedFnSymbol.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.IMPORT, Usage.Kind.OTHER);

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "import mod",
      "mod.fn(1, 2)"
    );
    callExpression = PythonTestUtils.getFirstChild(tree, t -> t.is(Tree.Kind.CALL_EXPR));
    importedFnSymbol = callExpression.calleeSymbol();
    assertThat(importedFnSymbol).isNotEqualTo(fnSymbol);
    assertThat(importedFnSymbol.kind()).isEqualTo(Symbol.Kind.FUNCTION);
    assertThat(fnSymbol.usages()).isEmpty();
    assertThat(importedFnSymbol.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.OTHER);

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "import mod as mod1",
      "mod1.fn(1, 2)"
    );
    callExpression = PythonTestUtils.getFirstChild(tree, t -> t.is(Tree.Kind.CALL_EXPR));
    importedFnSymbol = callExpression.calleeSymbol();
    assertThat(importedFnSymbol.kind()).isEqualTo(Symbol.Kind.FUNCTION);
    assertThat(importedFnSymbol.fullyQualifiedName()).isEqualTo("mod.fn");
    assertThat(importedFnSymbol.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.OTHER);
  }

  @Test
  public void import_already_existing_symbol() {
    FunctionDef functionDef = (FunctionDef) parse("def fn(p1, p2): pass").statements().statements().get(0);
    FunctionSymbolImpl fnSymbol = new FunctionSymbolImpl(functionDef, "mod.fn", pythonFile("mod.py"));
    List<Symbol> modSymbols = Collections.singletonList(fnSymbol);
    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("mod", new HashSet<>(modSymbols));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "fn = 42",
      "from mod import fn"
    );
    assertThat(tree.globalVariables()).hasSize(1);
    Symbol importedFnSymbol = tree.globalVariables().iterator().next();
    assertThat(importedFnSymbol.kind()).isEqualTo(Symbol.Kind.OTHER);
    assertThat(importedFnSymbol.name()).isEqualTo("fn");
    assertThat(importedFnSymbol.fullyQualifiedName()).isEqualTo(null);

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "mod = 42",
      "import mod"
    );
    assertThat(tree.globalVariables()).hasSize(1);
    Symbol modSymbol = tree.globalVariables().iterator().next();
    assertThat(modSymbol.kind()).isEqualTo(Symbol.Kind.OTHER);
    assertThat(modSymbol.name()).isEqualTo("mod");
    assertThat(modSymbol.fullyQualifiedName()).isEqualTo(null);
  }

  @Test
  public void other_imported_symbol() {
    SymbolImpl xSymbol = new SymbolImpl("x", "mod.x");
    List<Symbol> modSymbols = Collections.singletonList(xSymbol);
    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("mod", new HashSet<>(modSymbols));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "from mod import x"
    );
    Symbol importedXSymbol = tree.globalVariables().iterator().next();
    assertThat(importedXSymbol.name()).isEqualTo("x");
    assertThat(importedXSymbol.kind()).isEqualTo(Symbol.Kind.OTHER);
    assertThat(importedXSymbol.fullyQualifiedName()).isEqualTo("mod.x");
    assertThat(importedXSymbol.usages()).hasSize(1);
    assertThat(xSymbol).isNotEqualTo(importedXSymbol);
    assertThat(xSymbol.usages()).isEmpty();
  }

  @Test
  public void aliased_imported_symbols() {
    SymbolImpl xSymbol = new SymbolImpl("x", "mod.x");
    List<Symbol> modSymbols = Collections.singletonList(xSymbol);
    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("mod", new HashSet<>(modSymbols));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "from mod import x as y"
    );
    Symbol importedYSymbol = tree.globalVariables().iterator().next();
    assertThat(importedYSymbol.name()).isEqualTo("y");

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "import mod as mod1"
    );
    Symbol importedModSymbol = tree.globalVariables().iterator().next();
    assertThat(importedModSymbol.name()).isEqualTo("mod1");
  }

  @Test
  public void type_hierarchy() {
    ClassSymbolImpl classASymbol = new ClassSymbolImpl("A", "mod1.A");
    classASymbol.addSuperClass(new SymbolImpl("B", "mod2.B"));
    Set<Symbol> mod1Symbols = Collections.singleton(classASymbol);
    ClassSymbolImpl classBSymbol = new ClassSymbolImpl("B", "mod2.B");
    Set<Symbol> mod2Symbols = Collections.singleton(classBSymbol);
    Map<String, Set<Symbol>> globalSymbols = new HashMap<>();
    globalSymbols.put("mod1", mod1Symbols);
    globalSymbols.put("mod2", mod2Symbols);
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "from mod1 import A"
    );
    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.name()).isEqualTo("A");
    assertThat(importedASymbol.fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbol classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isEqualTo(false);
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "import mod1",
      "mod1.A"
    );

    QualifiedExpression qualifiedExpression = PythonTestUtils.getFirstChild(tree, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
    importedASymbol = qualifiedExpression.name().symbol();
    assertThat(importedASymbol.name()).isEqualTo("A");
    assertThat(importedASymbol.fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isEqualTo(false);
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);
  }

  @Test
  public void not_class_symbol_in_super_class() {
    ClassSymbolImpl classASymbol = new ClassSymbolImpl("A", "mod1.A");
    classASymbol.addSuperClass(new SymbolImpl("foo", "mod1.foo"));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(Collections.singletonMap("mod1", Collections.singleton(classASymbol)))),
      "from mod1 import A"
    );

    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbol classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isEqualTo(true);
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.OTHER);
  }

  @Test
  public void metaclass_in_imported_symbol() {
    Set<Symbol> globalsMod = parse(
      new SymbolTableBuilder("", pythonFile("mod1")),
      "from abc import ABCMeta",
      "class A(metaclass=ABCMeta): ..."
    ).globalVariables();
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(Collections.singletonMap("mod1", globalsMod))),
      "from mod1 import A"
    );

    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbolImpl classA = (ClassSymbolImpl) importedASymbol;
    assertThat(classA.hasMetaClass()).isTrue();
    assertThat(classA.metaclassFQN()).isEqualTo("abc.ABCMeta");
  }

  @Test
  public void builtin_symbol_in_super_class() {
    ClassSymbolImpl classASymbol = new ClassSymbolImpl("A", "mod1.A");
    classASymbol.addSuperClass(new SymbolImpl("BaseException", "BaseException"));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(Collections.singletonMap("mod1", Collections.singleton(classASymbol)))),
      "from mod1 import A"
    );

    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbol classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isEqualTo(true);
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.OTHER);
  }

  @Test
  public void multi_level_type_hierarchy() {
    ClassSymbolImpl classASymbol = new ClassSymbolImpl("A", "mod1.A");
    classASymbol.addSuperClass(new SymbolImpl("B", "mod2.B"));
    Set<Symbol> mod1Symbols = Collections.singleton(classASymbol);

    ClassSymbolImpl classBSymbol = new ClassSymbolImpl("B", "mod2.B");
    classBSymbol.addSuperClass(new SymbolImpl("C", "mod3.C"));
    Set<Symbol> mod2Symbols = Collections.singleton(classBSymbol);

    ClassSymbolImpl classCSymbol = new ClassSymbolImpl("C", "mod3.C");
    Set<Symbol> mod3Symbols = Collections.singleton(classCSymbol);

    Map<String, Set<Symbol>> globalSymbols = new HashMap<>();
    globalSymbols.put("mod1", mod1Symbols);
    globalSymbols.put("mod2", mod2Symbols);
    globalSymbols.put("mod3", mod3Symbols);
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "from mod1 import A"
    );
    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.name()).isEqualTo("A");
    assertThat(importedASymbol.fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbol classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isEqualTo(false);
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "import mod1",
      "mod1.A"
    );

    QualifiedExpression qualifiedExpression = PythonTestUtils.getFirstChild(tree, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
    importedASymbol = qualifiedExpression.name().symbol();
    assertThat(importedASymbol.name()).isEqualTo("A");
    assertThat(importedASymbol.fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isEqualTo(false);
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);

    ClassSymbol classB = (ClassSymbol) classA.superClasses().get(0);
    assertThat(classB.superClasses()).hasSize(1);
    assertThat(classB.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);
  }

  @Test
  public void ambiguous_imported_symbol() {
    Set<Symbol> modSymbols = parse(
      new SymbolTableBuilder("", pythonFile("mod")),
      "@overload",
      "def foo(a, b): ...",
      "@overload",
      "def foo(a, b, c): ..."
    ).globalVariables();

    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("mod", modSymbols);
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "from mod import foo"
    );
    Symbol importedFooSymbol = tree.globalVariables().iterator().next();
    assertThat(importedFooSymbol.name()).isEqualTo("foo");
    assertThat(importedFooSymbol.kind()).isEqualTo(Symbol.Kind.AMBIGUOUS);
    assertThat(importedFooSymbol.fullyQualifiedName()).isEqualTo("mod.foo");
    assertThat(importedFooSymbol.usages()).hasSize(1);
  }

  @Test
  public void imported_class_hasSuperClassWithoutSymbol() {
    Set<Symbol> modSymbols = parse(
      new SymbolTableBuilder("", pythonFile("mod")),
      "def foo(): ...",
      "class A(foo()): ..."
    ).globalVariables();
    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("mod", modSymbols);
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalSymbols)),
      "from mod import A"
    );
    Symbol importedFooSymbol = tree.globalVariables().iterator().next();
    assertThat(importedFooSymbol.name()).isEqualTo("A");
    assertThat(importedFooSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(importedFooSymbol.fullyQualifiedName()).isEqualTo("mod.A");
    ClassSymbol classSymbol = (ClassSymbol) importedFooSymbol;
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  private static Set<Symbol> globalSymbols(FileInput fileInput, String packageName) {
    ProjectLevelSymbolTable projectLevelSymbolTable = new ProjectLevelSymbolTable();
    projectLevelSymbolTable.addModule(fileInput, packageName, pythonFile("mod.py"));
    return projectLevelSymbolTable.getSymbolsFromModule(packageName.isEmpty() ? "mod" : packageName + ".mod");
  }

  @Test
  public void global_symbols() {
    FileInput tree = parseWithoutSymbols(
      "obj1 = 42",
      "obj2: int = 42",
      "def fn(): pass",
      "class A: pass"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("obj1", "obj2", "fn", "A");
    assertThat(globalSymbols).extracting(Symbol::fullyQualifiedName).containsExactlyInAnyOrder("mod.obj1", "mod.obj2", "mod.fn", "mod.A");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  public void global_symbols_private_by_convention() {
    // although being private by convention, it's considered as exported
    FileInput tree = parseWithoutSymbols(
      "def _private_fn(): pass"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("_private_fn");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  public void local_symbols_not_exported() {
    FileInput tree = parseWithoutSymbols(
      "def fn():",
      "  def inner(): pass",
      "  class Inner_class: pass",
      "class A:",
      "  def meth(): pass"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "mod");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("fn", "A");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  public void redefined_symbols() {
    FileInput tree = parseWithoutSymbols(
      "def fn(): pass",
      "def fn(): ...",
      "if True:",
      "  conditionally_defined = 1",
      "else:",
      "  conditionally_defined = 2"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "mod");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("fn", "conditionally_defined");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  public void function_symbols() {
    FileInput tree = parseWithoutSymbols(
      "def fn(): pass"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "mod");
    assertThat(globalSymbols).extracting(Symbol::kind).containsExactly(Symbol.Kind.FUNCTION);

    tree = parseWithoutSymbols(
      "def fn(): pass",
      "fn = 42"
    );
    globalSymbols = globalSymbols(tree, "mod");
    assertThat(globalSymbols).extracting(Symbol::kind).containsExactly(Symbol.Kind.OTHER);
  }

  @Test
  public void redefined_class_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "C = \"hello\"",
      "class C: ",
      "  pass");
    Set<Symbol> globalSymbols = globalSymbols(fileInput, "mod");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("C");
    assertThat(globalSymbols).extracting(Symbol::kind).allSatisfy(k -> assertThat(Symbol.Kind.CLASS.equals(k)).isFalse());
  }

  @Test
  public void classdef_with_missing_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "class C: ",
      "  pass",
      "global C");

    Set<Symbol> globalSymbols = globalSymbols(fileInput, "mod");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("C");
    // TODO: Global statements should not alter the kind of a symbol
    assertThat(globalSymbols).extracting(Symbol::kind).allSatisfy(k -> assertThat(Symbol.Kind.OTHER.equals(k)).isTrue());
  }

  @Test
  public void class_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "class C: ",
      "  pass");
    Set<Symbol> globalSymbols = globalSymbols(fileInput, "mod");
    assertThat(globalSymbols).hasSize(1);
    Symbol cSymbol = globalSymbols.iterator().next();
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).isEmpty();

    fileInput = parseWithoutSymbols(
      "class A: pass",
      "class C(A): ",
      "  pass");
    globalSymbols = globalSymbols(fileInput, "mod");
    Map<String, Symbol> symbols = globalSymbols.stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    cSymbol = symbols.get("C");
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).hasSize(1);

    // for the time being, we only consider symbols defined in the global scope
    fileInput = parseWithoutSymbols(
      "class A:",
      "  class A1: pass",
      "class C(A.A1): ",
      "  pass");
    globalSymbols = globalSymbols(fileInput, "mod");
    symbols = globalSymbols.stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    cSymbol = symbols.get("C");
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).hasSize(1);
  }

  @Test
  public void class_inheriting_from_imported_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "from mod import A",
      "import mod2",
      "class C(A): ",
      "  pass",
      "class D(mod2.B):",
      "  pass");

    Set<Symbol> globalSymbols = globalSymbols(fileInput, "mod");
    Map<String, Symbol> symbols = globalSymbols.stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    Symbol cSymbol = symbols.get("C");
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).hasSize(1);
    assertThat(((ClassSymbol) cSymbol).superClasses().get(0).fullyQualifiedName()).isEqualTo("mod.A");
    Symbol dSymbol = symbols.get("D");
    assertThat(dSymbol.name()).isEqualTo("D");
    assertThat(dSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) dSymbol).superClasses()).hasSize(1);
    assertThat(((ClassSymbol) dSymbol).superClasses().get(0).fullyQualifiedName()).isEqualTo("mod2.B");
  }

  @Test
  public void symbol_duplicated_by_wildcard_import() {
    FileInput tree = parseWithoutSymbols(
      "def nlargest(n, iterable): ...",
      "from _heapq import *",
      "def nlargest(n, iterable, key=None): ..."
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).isEmpty();

    tree = parseWithoutSymbols(
      "nonlocal nlargest",
      "from _heapq import *",
      "def nlargest(n, iterable, key=None): ..."
    );
    globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  public void class_having_itself_as_superclass_should_not_trigger_error() {
    FileInput fileInput = parseWithoutSymbols("class A(A): pass");
    Set<Symbol> globalSymbols = globalSymbols(fileInput, "mod");
    ClassSymbol a = (ClassSymbol) globalSymbols.iterator().next();
    assertThat(a.superClasses()).containsExactly(a);
  }

  @Test
  public void global_symbols_stdlib_imports() {
    FileInput tree = parseWithoutSymbols(
      "from time import time",
      "from threading import Thread",
      "from datetime import TimezoneMixin as tz",
      "import unknown",
      "from mod import *"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  public void module_importing_itself() {
    FileInput tree = parseWithoutSymbols(
      "from mod import *",
      "from mod import smth"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  public void imports_wont_trigger_typeshed_lookup() {
    // Imports from the file with the same name and with null name won't trigger TypeShed.symbolWithFQN
    FileInput tree = parse(
      new SymbolTableBuilder("", pythonFile("os.py"), empty()),
      "from os import *",
      "from . import *"
    );
    ImportFrom importFrom = ((ImportFrom) PythonTestUtils.getAllDescendant(tree, t -> t.is(Tree.Kind.IMPORT_FROM)).get(0));
    assertThat(importFrom.hasUnresolvedWildcardImport()).isTrue();
  }

  @Test
  public void loop_in_class_inheritance() {
    String[] foo = {
      "from bar import B",
      "class A(B): ..."
    };
    String[] bar = {
      "from foo import A",
      "class B(A): ..."
    };

    ProjectLevelSymbolTable projectSymbolTable = new ProjectLevelSymbolTable();
    projectSymbolTable.addModule(parseWithoutSymbols(foo), "", pythonFile("foo.py"));
    projectSymbolTable.addModule(parseWithoutSymbols(bar), "", pythonFile("bar.py"));

    Map<String, Symbol> barSymbols = getSymbolByName(parse(new SymbolTableBuilder("", pythonFile("bar.py"), projectSymbolTable), bar));

    ClassSymbol classB = (ClassSymbol) barSymbols.get("B");
    assertThat(classB.superClasses())
      .extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactly(tuple(Symbol.Kind.CLASS, "foo.A"));

    ClassSymbolImpl importedA = (ClassSymbolImpl) barSymbols.get("A");
    assertThat(importedA.superClasses())
      .extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactly(tuple(Symbol.Kind.CLASS, "bar.B"));

    // at this level, parent of imported class is not Class Symbol anymore because of the loop in  class inheritance
    ClassSymbolImpl parentOfImportedA = (ClassSymbolImpl) importedA.superClasses().iterator().next();
    assertThat(parentOfImportedA.superClasses())
      .extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactly(tuple(Symbol.Kind.OTHER, "foo.A"));
  }

  @Test
  public void django_views() {
    String[] urls = {
      "from django.urls import path, other",
      "import views",
      "urlpatterns = [path('foo', views.foo, name='foo'), path('baz')]",
      "other(views.bar)",
      "unknown()",
    };
    String[] views = {
      "def foo(): ...",
      "def bar(): ..."
    };

    ProjectLevelSymbolTable projectSymbolTable = new ProjectLevelSymbolTable();
    projectSymbolTable.addModule(parseWithoutSymbols(urls), "", pythonFile("urls.py"));

    assertThat(projectSymbolTable.isDjangoView("views.foo")).isTrue();

    FileInput fileInput = parse(new SymbolTableBuilder("", pythonFile("views.py"), projectSymbolTable), views);
    Map<String, Symbol> symbolByName = getSymbolByName(fileInput);
    FunctionSymbolImpl foo = ((FunctionSymbolImpl) symbolByName.get("foo"));
    FunctionSymbolImpl bar = ((FunctionSymbolImpl) symbolByName.get("bar"));
    assertThat(foo.isDjangoView()).isTrue();
    assertThat(bar.isDjangoView()).isFalse();
  }

}
