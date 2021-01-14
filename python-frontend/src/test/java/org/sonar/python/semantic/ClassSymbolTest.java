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

import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Statement;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.groups.Tuple.tuple;
import static org.sonar.python.PythonTestUtils.parse;

public class ClassSymbolTest {

  @Test
  public void no_parents() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(0);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();

    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(0);
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isFalse();
  }

  @Test
  public void local_parent() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "class B(C): ",
      "  pass");
    ClassDef parentClass = (ClassDef) fileInput.statements().statements().get(0);
    Symbol parentSymbol = parentClass.name().symbol();
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(1);
    assertThat(classSymbol.superClasses()).containsExactlyInAnyOrder(parentSymbol);

    assertThat(fileInput.globalVariables()).hasSize(2);
    assertThat(fileInput.globalVariables()).containsExactlyInAnyOrder(symbol, parentSymbol);
  }

  @Test
  public void multiple_local_parents() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "class A:",
      "  pass",
      "class B(C, A): ",
      "  pass");
    ClassDef parentClass = (ClassDef) fileInput.statements().statements().get(0);
    Symbol parentSymbol = parentClass.name().symbol();
    ClassDef parentClass2 = (ClassDef) fileInput.statements().statements().get(1);
    Symbol parentSymbol2 = parentClass2.name().symbol();
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(2);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(2);
    assertThat(classSymbol.superClasses()).containsExactlyInAnyOrder(parentSymbol, parentSymbol2);

    assertThat(fileInput.globalVariables()).hasSize(3);
    assertThat(fileInput.globalVariables()).containsExactlyInAnyOrder(symbol, parentSymbol, parentSymbol2);
  }

  @Test
  public void unknown_parent() {
    FileInput fileInput = parse(
      "class B(C): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(0);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(0);
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  public void builtin_parent() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "class B(C, BaseException): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(2);
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isFalse();
  }

  @Test
  public void builtin_parent_with_unknown() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "class B(C, BaseException, unknown): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(2);
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  public void multiple_bindings() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "C = \"hello\"");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(0);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isFalse();
    assertThat(symbol.kind().equals(Symbol.Kind.AMBIGUOUS)).isTrue();
  }

  @Test
  public void multiple_bindings_2() {
    FileInput fileInput = parse(
      "C = \"hello\"",
      "class C: ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isFalse();
    assertThat(Symbol.Kind.CLASS.equals(symbol.kind())).isFalse();
  }

  @Test
  public void call_expression_argument() {
    FileInput fileInput = parse(
      "def foo():",
      "  pass",
      "class C(foo()): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(0);
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  public void parent_is_not_a_class() {
    FileInput fileInput = parse(
      "def foo():",
      "  pass",
      "A = foo()",
      "class C(A): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(2);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(1);
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  public void unpacking_expression_as_parent() {
    FileInput fileInput = parse(
      "foo = (Something, SomethingElse)",
      "class C(*foo): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.superClasses()).hasSize(0);
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  public void parent_has_multiple_bindings() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "C = \"hello\"",
      "class B(C): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(2);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  public void defines_metaclass() {
    FileInput fileInput = parse(
      "class A: ",
      "  pass",
      "class B(metaclass=A): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol).isInstanceOf(ClassSymbol.class);
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(classSymbol.superClasses()).isEmpty();
    assertThat(classSymbol.hasMetaClass()).isTrue();
    assertThat(classSymbol.canHaveMember("foo")).isTrue();

    ClassSymbol C = lastClassSymbol(
      "class A: ",
      "  pass",
      "class B(metaclass=A): ",
      "  pass",
      "class C(B):",
      "  pass");

    assertThat(C.canHaveMember("foo")).isTrue();

    C = lastClassSymbol(
      "from abc import ABCMeta",
      "class B(metaclass=ABCMeta): ",
      "  pass",
      "class C(B):",
      "  pass");

    assertThat(C.canHaveMember("foo")).isFalse();

    C = lastClassSymbol(
      "from abc import ABCMeta",
      "class Factory: ...",
      "class A(metaclass=Factory): ",
      "  pass",
      "class B(A, metaclass=ABCMeta): ",
      "  pass",
      "class C(B):",
      "  pass");

    assertThat(C.canHaveMember("foo")).isTrue();
  }

  @Test
  public void defines_metaclass_python_2() {
    FileInput fileInput = parse(
      "class A: ",
      "  pass",
      "class B(): ",
      "  __metaclass__ = A");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol).isInstanceOf(ClassSymbol.class);
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(classSymbol.superClasses()).isEmpty();
    assertThat(classSymbol.hasMetaClass()).isTrue();
    assertThat(classSymbol.canHaveMember("foo")).isTrue();
  }

  @Test
  public void defines_attrs() {
    FileInput fileInput = parse(
      "class A: ",
      "  pass",
      "class B(A, attrs=...): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol).isInstanceOf(ClassSymbol.class);
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
    assertThat(classSymbol.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(classSymbol.superClasses()).hasSize(1);
    assertThat(classSymbol.superClasses()).extracting(Symbol::name).containsExactly("A");
    assertThat(classSymbol.hasMetaClass()).isFalse();
    assertThat(classSymbol.canHaveMember("foo")).isFalse();
  }

  @Test
  public void class_with_global_statement() {
    FileInput fileInput = parse(
      "global B",
      "class B(): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    //Currently, no symbol is created in case of global symbol
    assertThat(symbol instanceof ClassSymbol).isFalse();
  }

  @Test
  public void class_with_nonlocal_statement() {
    FileInput fileInput = parse(
      "nonlocal B",
      "class B(): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    //Currently, no symbol is created in case of nonlocal symbol
    assertThat(symbol instanceof ClassSymbol).isFalse();
  }

  @Test
  public void class_members_empty() {
    ClassSymbol symbol = lastClassSymbol(
      "class C: ",
      "  pass");
    assertThat(symbol.declaredMembers()).isEmpty();
  }

  @Test
  public void class_members() {
    ClassSymbol symbol = lastClassSymbol(
      "class C: ",
      "  def foo(): pass");
    assertThat(symbol.declaredMembers()).extracting("kind", "name").containsExactlyInAnyOrder(tuple(Symbol.Kind.FUNCTION, "foo"));

    symbol = lastClassSymbol(
      "class C: ",
      "  bar = 42");
    assertThat(symbol.declaredMembers()).extracting("kind", "name").containsExactlyInAnyOrder(tuple(Symbol.Kind.OTHER, "bar"));
  }

  @Test
  public void duplicated_class_member_self() {
    ClassSymbol symbol = lastClassSymbol(
      "class C:",
      "  def f(self): ...",
      "  def g(self): ",
      "    self.f()"
    );
    assertThat(symbol.declaredMembers()).extracting(Symbol::name, Symbol::kind).containsExactlyInAnyOrder(tuple("f", Symbol.Kind.FUNCTION), tuple("g", Symbol.Kind.FUNCTION));

    ClassSymbol classSymbol = lastClassSymbol(
      "class A:",
      "  def __init__(self):",
      "    self.foo = []",
      "  def foo(self): ..."
    );
    assertThat(classSymbol.declaredMembers()).extracting(Symbol::name, Symbol::kind)
      .containsExactlyInAnyOrder(tuple("foo", Symbol.Kind.OTHER), tuple("__init__", Symbol.Kind.FUNCTION));
  }

  @Test
  public void class_members_with_inheritance() {
    ClassSymbol symbol = lastClassSymbol(
      "class A:",
      "  def meth(): pass",
      "class B(A): ",
      "  def foo(): pass");

    assertThat(symbol.declaredMembers()).extracting("kind", "name").containsExactlyInAnyOrder(tuple(Symbol.Kind.FUNCTION, "foo"));
    ClassSymbol classA = ((ClassSymbol) symbol.superClasses().get(0));
    assertThat(classA.declaredMembers()).extracting("kind", "name").containsExactlyInAnyOrder(tuple(Symbol.Kind.FUNCTION, "meth"));
  }

  @Test
  public void copy_without_usages() {
    ClassSymbolImpl classSymbol = ((ClassSymbolImpl) lastClassSymbol(
      "class A(foo):",
      "  def meth(): pass"));

    assertEqualsWithoutUsages(classSymbol);

    classSymbol = ((ClassSymbolImpl) lastClassSymbol(
      "class A:",
      "  def meth(): pass",
      "class B(A): ",
      "  def foo(): pass"));

    assertEqualsWithoutUsages(classSymbol);

    ClassSymbolImpl classSymbolWithAmbiguousParent = new ClassSymbolImpl("B", "foo.B");
    AmbiguousSymbol ambiguousParent = AmbiguousSymbolImpl.create(new SymbolImpl("x", "foo.x"), new SymbolImpl("x", "bar.x"));
    classSymbolWithAmbiguousParent.addSuperClass(ambiguousParent);

    assertEqualsWithoutUsages(classSymbolWithAmbiguousParent);
  }

  @Test
  public void static_member_usages() {
    ClassSymbol classSymbol = lastClassSymbol(
            "class A:",
            "  foo = 42",
            "  def __init__(self): ",
            "    A.foo",
            "    A.foo = 0",
            "    A.bar"
    );
    Symbol foo = classSymbol.resolveMember("foo").get();
    assertThat(foo.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.ASSIGNMENT_LHS,  Usage.Kind.OTHER, Usage.Kind.ASSIGNMENT_LHS);
    assertThat(classSymbol.resolveMember("bar")).isEmpty();
  }

  @Test
  public void inherited_static_member() {
    ClassSymbol classSymbol = firstClassSymbol(
            "class A:",
            "  foo = 42",
            "class B(A): pass",
            "B.foo"
    );

    assertThat(classSymbol.canHaveMember("foo")).isTrue();
    Symbol foo = classSymbol.resolveMember("foo").get();
    assertThat(foo.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.OTHER);
  }

  @Test
  public void inherits_from_ambiguous_symbol() {
    ClassSymbol classSymbol = lastClassSymbol(
      "if x:",
      "  class A: ...",
      "else:",
      "  class A:",
      "    def foo(): ...",
      "class B(A): ..."
    );

    assertThat(classSymbol.resolveMember("foo")).isNotPresent();
    assertThat(classSymbol.canHaveMember("foo")).isTrue();
  }

  @Test
  public void inherits_from_function_call() {
    ClassSymbol classSymbol = lastClassSymbol(
      "class A:",
      "  def foo(): ...",
      "def func(): return A",
      "class B(func()): ..."
    );

    assertThat(classSymbol.resolveMember("foo")).isNotPresent();
    assertThat(classSymbol.canHaveMember("foo")).isTrue();
  }

  @Test
  public void has_decorators() {
    ClassSymbol classSymbol = firstClassSymbol(
      "@foo",
      "class A: ..."
    );
    assertThat(classSymbol.hasDecorators()).isTrue();

    classSymbol = firstClassSymbol(
      "class A: ..."
    );
    assertThat(classSymbol.hasDecorators()).isFalse();
  }

  private static void assertEqualsWithoutUsages(ClassSymbolImpl classSymbol) {
    ClassSymbolImpl copied = classSymbol.copyWithoutUsages();
    assertThat(copied.hasUnresolvedTypeHierarchy()).isEqualTo(classSymbol.hasUnresolvedTypeHierarchy());

    List<String> copiedfqnSuperClasses = copied.superClasses().stream().map(Symbol::fullyQualifiedName).collect(Collectors.toList());
    List<String> fqnSuperClasses = classSymbol.superClasses().stream().map(Symbol::fullyQualifiedName).collect(Collectors.toList());
    assertThat(copiedfqnSuperClasses).isEqualTo(fqnSuperClasses);

    List<Symbol.Kind> copiedKindSuperClasses = copied.superClasses().stream().map(Symbol::kind).collect(Collectors.toList());
    List<Symbol.Kind> kindSuperClasses = classSymbol.superClasses().stream().map(Symbol::kind).collect(Collectors.toList());
    assertThat(copiedKindSuperClasses).isEqualTo(kindSuperClasses);

    List<String> copiedFqnMembers = copied.declaredMembers().stream().map(Symbol::fullyQualifiedName).collect(Collectors.toList());
    List<String> fqnMembers = classSymbol.declaredMembers().stream().map(Symbol::fullyQualifiedName).collect(Collectors.toList());
    assertThat(copiedFqnMembers).isEqualTo(fqnMembers);

    List<Symbol.Kind> copiedKindMembers = copied.declaredMembers().stream().map(Symbol::kind).collect(Collectors.toList());
    List<Symbol.Kind> kindMembers = classSymbol.declaredMembers().stream().map(Symbol::kind).collect(Collectors.toList());
    assertThat(copiedKindMembers).isEqualTo(kindMembers);

    assertThat(copied.usages()).isEmpty();
  }


  private static ClassSymbol firstClassSymbol(String... code) {
    FileInput fileInput = parse(code);
    List<Statement> statements = fileInput.statements().statements();
    ClassDef classDef = (ClassDef) statements.get(0);
    return (ClassSymbol) classDef.name().symbol();
  }

  private static ClassSymbol lastClassSymbol(String... code) {
    FileInput fileInput = parse(code);
    List<Statement> statements = fileInput.statements().statements();
    ClassDef classDef = (ClassDef) statements.get(statements.size() - 1);
    return (ClassSymbol) classDef.name().symbol();
  }
}
