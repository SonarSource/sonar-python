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
package org.sonar.python.types;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.semantic.SymbolTableBuilder;

import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.types.DeclaredType.fromInferredType;
import static org.sonar.python.types.InferredTypes.DECL_INT;
import static org.sonar.python.types.InferredTypes.DECL_LIST;
import static org.sonar.python.types.InferredTypes.INT;
import static org.sonar.python.types.InferredTypes.STR;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.or;

class DeclaredTypeTest {

  private final ClassSymbolImpl a = new ClassSymbolImpl("a", "a");
  private final ClassSymbolImpl b = new ClassSymbolImpl("b", "b");
  private final ClassSymbolImpl c = new ClassSymbolImpl("c", "c");

  @Test
  void isIdentityComparableWith() {
    DeclaredType aType = new DeclaredType(a);
    DeclaredType bType = new DeclaredType(b);
    DeclaredType cType = new DeclaredType(c);

    assertThat(aType.isIdentityComparableWith(bType)).isTrue();
    assertThat(aType.isIdentityComparableWith(aType)).isTrue();
    assertThat(aType.isIdentityComparableWith(new RuntimeType(a))).isTrue();

    assertThat(aType.isIdentityComparableWith(AnyType.ANY)).isTrue();

    assertThat(aType.isIdentityComparableWith(or(aType, bType))).isTrue();
    assertThat(aType.isIdentityComparableWith(or(cType, bType))).isTrue();
  }

  @Test
  void member() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    SymbolImpl foo = new SymbolImpl("foo", null);
    x.addMembers(singletonList(foo));
    assertThat(new DeclaredType(x).canHaveMember("foo")).isTrue();
    assertThat(new DeclaredType(x).canHaveMember("bar")).isTrue();
    assertThat(new DeclaredType(x).resolveMember("foo")).isEmpty();
    assertThat(new DeclaredType(x).resolveMember("bar")).isEmpty();

    ClassSymbol classSymbol = lastClassSymbol(
      "class C:",
      "  def foo(): ..."
    );
    DeclaredType declaredType = new DeclaredType(classSymbol);
    assertThat(declaredType.declaresMember("foo")).isTrue();
    assertThat(declaredType.declaresMember("bar")).isFalse();

    DeclaredType emptyUnion = new DeclaredType(new SymbolImpl("Union", "typing.Union"));
    assertThat(emptyUnion.declaresMember("foo")).isTrue();

    classSymbol = lastClassSymbol(
      "class Base:",
      "  def bar(): ...",
      "class C(Base):",
      "  def foo(): ..."
    );
    declaredType = new DeclaredType(classSymbol);
    assertThat(declaredType.declaresMember("foo")).isTrue();
    assertThat(declaredType.declaresMember("bar")).isTrue();
    assertThat(declaredType.declaresMember("other")).isFalse();

    assertThat(new DeclaredType(new SymbolImpl("x", "foo.x")).declaresMember("member")).isTrue();

    ClassSymbolImpl noFullyQualifiedName = new ClassSymbolImpl("unknown", null);
    noFullyQualifiedName.addMembers(singletonList(foo));
    assertThat(new DeclaredType(noFullyQualifiedName).canHaveMember("foo")).isTrue();
    assertThat(new DeclaredType(noFullyQualifiedName).canHaveMember("bar")).isTrue();
    assertThat(new DeclaredType(noFullyQualifiedName).declaresMember("foo")).isTrue();
    assertThat(new DeclaredType(noFullyQualifiedName).declaresMember("bar")).isFalse();
    assertThat(new DeclaredType(noFullyQualifiedName).resolveMember("foo")).isEmpty();
    assertThat(new DeclaredType(noFullyQualifiedName).resolveMember("bar")).isEmpty();
  }

  @Test
  void mocks_can_have_and_declare_any_members() {
    ClassSymbolImpl x = new ClassSymbolImpl("Mock", "unittest.mock.Mock");
    DeclaredType declaredType = new DeclaredType(x);
    assertThat(declaredType.canHaveMember("foo")).isTrue();
    assertThat(declaredType.canHaveMember("bar")).isTrue();
    assertThat(declaredType.declaresMember("foo")).isTrue();
    assertThat(declaredType.declaresMember("bar")).isTrue();
    assertThat(declaredType.resolveMember("foo")).isEmpty();
    assertThat(declaredType.resolveMember("bar")).isEmpty();

    ClassSymbolImpl magicMock = new ClassSymbolImpl("MagicMock", "unittest.mock.MagicMock");
    DeclaredType declaredTypeMagicMock = new DeclaredType(magicMock);
    assertThat(declaredTypeMagicMock.canHaveMember("bar")).isTrue();
    assertThat(declaredTypeMagicMock.declaresMember("foo")).isTrue();
    assertThat(declaredTypeMagicMock.resolveMember("foo")).isEmpty();
    assertThat(declaredTypeMagicMock.resolveMember("bar")).isEmpty();

    ClassSymbolImpl extendedMock = new ClassSymbolImpl("Extended", "Extended");
    extendedMock.addSuperClass(x);
    DeclaredType extendedType = new DeclaredType(extendedMock);
    assertThat(extendedType.canHaveMember("foo")).isTrue();
    assertThat(extendedType.canHaveMember("bar")).isTrue();
    assertThat(extendedType.canHaveMember("foobar")).isTrue();
    assertThat(extendedType.declaresMember("foo")).isTrue();
    assertThat(extendedType.declaresMember("bar")).isTrue();
    assertThat(extendedType.declaresMember("foobar")).isTrue();
    assertThat(extendedType.resolveMember("foo")).isEmpty();
    assertThat(extendedType.resolveMember("bar")).isEmpty();
    assertThat(extendedType.resolveMember("foobar")).isEmpty();
  }

  @Test
  void test_toString() {
    assertThat(new DeclaredType(a)).hasToString("DeclaredType(a)");
    assertThat(new DeclaredType(a, Collections.singletonList(new DeclaredType(b)))).hasToString("DeclaredType(a[b])");
    assertThat(new DeclaredType(a, Arrays.asList(new DeclaredType(b), new DeclaredType(c)))).hasToString("DeclaredType(a[b, c])");
  }

  @Test
  void test_canOnlyBe() {
    assertThat(new DeclaredType(a).canOnlyBe("a")).isFalse();
    assertThat(new DeclaredType(b).canOnlyBe("a")).isFalse();
  }

  @Test
  void test_canBeOrExtend() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    assertThat(new DeclaredType(x).canBeOrExtend("x")).isTrue();
    assertThat(new DeclaredType(x).canBeOrExtend("y")).isTrue();
  }

  @Test
  void test_isCompatibleWith() {
    ClassSymbol x1 = lastClassSymbol(
      "class X1:",
      "  def foo(): ..."
    );
    ClassSymbol x2 = lastClassSymbol(
      "class X2:",
      "  def bar(): ..."
    );
    ((ClassSymbolImpl) x2).addSuperClass(x1);

    assertThat(new DeclaredType(x2).isCompatibleWith(new DeclaredType(x1))).isTrue();
    assertThat(new DeclaredType(x1).isCompatibleWith(new DeclaredType(x1))).isTrue();
    assertThat(new DeclaredType(x1).isCompatibleWith(new DeclaredType(x2))).isFalse();
    assertThat(new DeclaredType(x1).isCompatibleWith(INT)).isFalse();
    DeclaredType emptyUnion = new DeclaredType(new SymbolImpl("Union", "typing.Union"));
    assertThat(emptyUnion.isCompatibleWith(INT)).isTrue();
  }

  @Test
  void test_mustBeOrExtend() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addSuperClass(x1);

    DeclaredType typeX1 = new DeclaredType(x1);
    DeclaredType typeX2 = new DeclaredType(x2);

    assertThat(typeX1.mustBeOrExtend("x1")).isTrue();
    assertThat(typeX1.mustBeOrExtend("x2")).isFalse();

    assertThat(typeX2.mustBeOrExtend("x1")).isTrue();
    assertThat(typeX2.mustBeOrExtend("x2")).isTrue();

    ClassSymbolImpl otherX1 = new ClassSymbolImpl("x1", "x1");
    Set<Symbol> symbols = new HashSet<>(Arrays.asList(x1, otherX1));
    AmbiguousSymbol ambiguousSymbol = new AmbiguousSymbolImpl("x1", "x1", symbols);
    DeclaredType typeAmbiguousX1 = new DeclaredType(ambiguousSymbol);
    assertThat(typeAmbiguousX1.mustBeOrExtend("x1")).isTrue();
    assertThat(typeAmbiguousX1.mustBeOrExtend("other")).isFalse();

    DeclaredType declaredType = new DeclaredType(new SymbolImpl("C", "foo.C"));
    assertThat(declaredType.mustBeOrExtend("other")).isFalse();
    assertThat(declaredType.mustBeOrExtend("foo.C")).isFalse();
    assertThat(declaredType.mustBeOrExtend("C")).isFalse();
  }

  @Test
  void test_getClass() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    assertThat(new DeclaredType(x1).getTypeClass()).isEqualTo(x1);
  }

  @Test
  void test_equals() {
    DeclaredType aType = new DeclaredType(a);
    assertThat(aType)
      .isEqualTo(aType)
      .isEqualTo(new DeclaredType(a))
      .isNotEqualTo(new DeclaredType(b))
      .isNotEqualTo(a)
      .isNotEqualTo(null)
      .isNotEqualTo(new DeclaredType(a, Arrays.asList(new DeclaredType(b), new DeclaredType(c))))
      .isEqualTo(new DeclaredType(new SymbolImpl("a", "a")))
      .isNotEqualTo(new DeclaredType(new SymbolImpl("a", "b")));

    DeclaredType x = new DeclaredType(new ClassSymbolImpl("X", null));
    DeclaredType y = new DeclaredType(new ClassSymbolImpl("Y", null));
    assertThat(x).isNotEqualTo(y);
  }

  @Test
  void test_hashCode() {
    DeclaredType aType = new DeclaredType(a);
    assertThat(aType.hashCode()).isEqualTo(new DeclaredType(a).hashCode());
    assertThat(aType.hashCode()).isNotEqualTo(new DeclaredType(b).hashCode());
    assertThat(aType.hashCode()).isNotEqualTo(new DeclaredType(a, Arrays.asList(new DeclaredType(b), new DeclaredType(c))).hashCode());

    DeclaredType x = new DeclaredType(new ClassSymbolImpl("X", null));
    DeclaredType y = new DeclaredType(new ClassSymbolImpl("Y", null));
    assertThat(x.hashCode()).isNotEqualTo(y.hashCode());
  }

  @Test
  void test_fromInferredType() {
    assertThat(fromInferredType(anyType())).isEqualTo(anyType());
    assertThat(fromInferredType(INT)).isEqualTo(DECL_INT);
    assertThat(fromInferredType(DECL_INT)).isEqualTo(DECL_INT);
    assertThat(fromInferredType(or(INT, STR))).isEqualTo(anyType());
  }

  @Test
  void test_resolveDeclaredMember() {
    ClassSymbolImpl typeClassX = new ClassSymbolImpl("x", "x");
    SymbolImpl fooX = new SymbolImpl("foo", "x.foo");
    typeClassX.addMembers(Collections.singletonList(fooX));
    DeclaredType declaredTypeX = new DeclaredType(typeClassX);
    assertThat(declaredTypeX.resolveDeclaredMember("foo")).contains(fooX);
    assertThat(declaredTypeX.resolveDeclaredMember("bar")).isEmpty();

    ClassSymbolImpl typeClassY = new ClassSymbolImpl("y", "y");
    SymbolImpl fooY = new SymbolImpl("foo", "y.foo");
    typeClassY.addMembers(Collections.singletonList(fooY));
    DeclaredType declaredTypeY = new DeclaredType(typeClassY);
    DeclaredType union = new DeclaredType(new SymbolImpl("Union", "typing.Union"), Arrays.asList(declaredTypeX, declaredTypeY));
    assertThat(union.resolveDeclaredMember("foo")).isEmpty();
    assertThat(union.resolveDeclaredMember("bar")).isEmpty();

    DeclaredType unresolved = new DeclaredType(new SymbolImpl("unresolved", "unresolved"));
    union = new DeclaredType(new SymbolImpl("Union", "typing.Union"), Arrays.asList(declaredTypeX, unresolved));
    assertThat(union.resolveDeclaredMember("foo")).isEmpty();
  }

  @Test
  void test_hasUnresolvedHierarchy() {
    ClassSymbolImpl typeClassX = new ClassSymbolImpl("x", "x");
    DeclaredType declaredTypeX = new DeclaredType(typeClassX);
    assertThat(declaredTypeX.hasUnresolvedHierarchy()).isFalse();
    DeclaredType union = new DeclaredType(new SymbolImpl("Union", "typing.Union"));
    assertThat(union.hasUnresolvedHierarchy()).isTrue();
    DeclaredType unresolved = new DeclaredType(new SymbolImpl("unresolved", "unresolved"));
    union = new DeclaredType(new SymbolImpl("Union", "typing.Union"), Arrays.asList(declaredTypeX, unresolved));
    assertThat(union.hasUnresolvedHierarchy()).isTrue();
  }

  @Test
  void test_generic_collections() {
    assertThat(lastExpression(
      "def f(x: list):",
      "  x"
    ).type()).isEqualTo(DECL_LIST);

    assertThat(((DeclaredType) lastExpression(
      "def f(x: list[int]):",
      "  x"
    ).type()).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName).containsExactly("list");
  }

  private static ClassSymbol lastClassSymbol(String... code) {
    FileInput fileInput = parse(new SymbolTableBuilder("my_package", pythonFile("my_module.py")), code);
    List<Statement> statements = fileInput.statements().statements();
    ClassDef classDef = (ClassDef) statements.get(statements.size() - 1);
    return (ClassSymbol) classDef.name().symbol();
  }
}
