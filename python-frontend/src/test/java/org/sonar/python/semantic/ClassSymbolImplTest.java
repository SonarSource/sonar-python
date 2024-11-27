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

import com.google.protobuf.TextFormat;
import java.util.Collections;
import java.util.HashSet;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.python.PythonTestUtils.parse;

class ClassSymbolImplTest {
  @BeforeEach
  void setup() {
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void hasUnresolvedTypeHierarchy() {
    ClassSymbolImpl a = new ClassSymbolImpl("x", null);
    assertThat(a.hasUnresolvedTypeHierarchy()).isFalse();

    ClassSymbolImpl b = new ClassSymbolImpl("x", null);
    b.addSuperClass(new ClassSymbolImpl("s", null));
    assertThat(b.hasUnresolvedTypeHierarchy()).isFalse();

    ClassSymbolImpl c = new ClassSymbolImpl("x", null);
    c.addSuperClass(new SymbolImpl("s", null));
    assertThat(c.hasUnresolvedTypeHierarchy()).isTrue();

    ClassSymbolImpl d = new ClassSymbolImpl("x", null);
    d.addSuperClass(c);
    assertThat(d.hasUnresolvedTypeHierarchy()).isTrue();

    ClassSymbolImpl e = new ClassSymbolImpl("x", null);
    e.setHasSuperClassWithoutSymbol();
    assertThat(e.hasUnresolvedTypeHierarchy()).isTrue();

    ClassSymbolImpl f = new ClassSymbolImpl("x", null);
    Symbol g = new SymbolImpl("g", null);
    f.addSuperClass(g);
    assertThat(f.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  void cycle_between_super_classes() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    ClassSymbolImpl z = new ClassSymbolImpl("z", "z");
    x.addSuperClass(y);
    y.addSuperClass(z);
    z.addSuperClass(x);
    assertThat(x.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(x.isOrExtends("y")).isTrue();
    assertThat(x.isOrExtends("a")).isFalse();
  }

  @Test
  void should_throw_when_adding_super_class_after_super_classes_were_read() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    ClassSymbolImpl b = new ClassSymbolImpl("b", null);
    ClassSymbolImpl c = new ClassSymbolImpl("c", null);
    a.addSuperClass(b);
    assertThat(a.superClasses()).containsExactly(b);
    assertThatThrownBy(() -> a.addSuperClass(c)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void should_throw_when_adding_super_class_after_checking_hierarchy() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    ClassSymbolImpl b = new ClassSymbolImpl("b", null);
    ClassSymbolImpl c = new ClassSymbolImpl("c", null);
    a.addSuperClass(b);
    assertThat(a.hasUnresolvedTypeHierarchy()).isFalse();
    assertThatThrownBy(() -> a.addSuperClass(c)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void resolveMember() {
    assertThat(new ClassSymbolImpl("a", null).resolveMember("foo")).isEmpty();

    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    Symbol fooA = new SymbolImpl("foo", "a.foo");
    a.addMembers(Collections.singleton(fooA));
    assertThat(a.resolveMember("foo")).contains(fooA);

    ClassSymbolImpl b = new ClassSymbolImpl("b", null);
    Symbol c = new SymbolImpl("c", null);
    b.addSuperClass(c);
    assertThat(b.resolveMember("foo")).isEmpty();
  }

  @Test
  void resolve_inherited_member() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    ClassSymbolImpl b = new ClassSymbolImpl("b", null);
    Symbol fooB = new SymbolImpl("foo", "b.foo");
    b.addMembers(Collections.singleton(fooB));
    a.addSuperClass(b);
    assertThat(a.resolveMember("foo")).contains(fooB);
  }

  @Test
  void resolve_overridden_member() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    Symbol fooA = new SymbolImpl("foo", "a.foo");
    a.addMembers(Collections.singleton(fooA));
    ClassSymbolImpl b = new ClassSymbolImpl("b", null);
    Symbol fooB = new SymbolImpl("foo", "b.foo");
    b.addMembers(Collections.singleton(fooB));
    a.addSuperClass(b);
    assertThat(a.resolveMember("foo")).contains(fooA);
  }

  @Test
  void should_throw_when_adding_member_after_call_to_resolveMember() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    a.addMembers(Collections.singleton(new SymbolImpl("m1", null)));
    assertThat(a.resolveMember("m1")).isNotEmpty();
    assertThatThrownBy(() -> a.addMembers(Collections.singleton(new SymbolImpl("m2", null)))).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void isOrExtends() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", "mod1.a");
    ClassSymbolImpl b = new ClassSymbolImpl("b", "mod2.b");
    a.addSuperClass(b);
    assertThat(a.isOrExtends("a")).isFalse();
    assertThat(a.isOrExtends("mod1.a")).isTrue();
    assertThat(a.isOrExtends("mod2.b")).isTrue();
    assertThat(a.isOrExtends("mod2.x")).isFalse();

    assertThat(a.isOrExtends(a)).isTrue();
    assertThat(a.isOrExtends(b)).isTrue();
    assertThat(b.isOrExtends(a)).isFalse();
    ClassSymbolImpl c = new ClassSymbolImpl("c", "mod2.c");
    assertThat(a.isOrExtends(c)).isFalse();

    assertThat(new ClassSymbolImpl("foo", "foo").isOrExtends(TypeShed.typeShedClass("object"))).isTrue();
    assertThat(a.isOrExtends(TypeShed.typeShedClass("object"))).isTrue();
  }

  @Test
  void isOrExtends_non_class_symbol() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", "mod1.a");
    SymbolImpl b = new SymbolImpl("b", "mod2.b");
    a.addSuperClass(b);

    assertThat(a.isOrExtends("mod2.b")).isTrue();

    ClassSymbolImpl c = new ClassSymbolImpl("c", "mod1.c");
    SymbolImpl d = new SymbolImpl("d", null);
    c.addSuperClass(d);

    assertThat(c.isOrExtends("d")).isFalse();
    assertThat(c.isOrExtends((String) null)).isFalse();
  }

  @Test
  void canBeOrExtend() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", "mod1.a");
    ClassSymbolImpl b = new ClassSymbolImpl("b", "mod2.b");
    a.addSuperClass(b);
    assertThat(a.canBeOrExtend("a")).isFalse();
    assertThat(a.canBeOrExtend("mod1.a")).isTrue();
    assertThat(a.canBeOrExtend("mod2.b")).isTrue();
    assertThat(a.canBeOrExtend("mod2.x")).isFalse();

    ClassSymbolImpl c = new ClassSymbolImpl("c", "mod1.c");
    HashSet<Symbol> alternatives = new HashSet<>();
    alternatives.add(a);
    alternatives.add(new ClassSymbolImpl("a", "mod2.a"));
    AmbiguousSymbol aOrB = AmbiguousSymbolImpl.create(alternatives);
    c.addSuperClass(aOrB);
    assertThat(c.canBeOrExtend("mod1.a")).isTrue();
    assertThat(c.isOrExtends("mod1.a")).isFalse();
    assertThat(c.canBeOrExtend("mod2.a")).isTrue();
    assertThat(c.canBeOrExtend("mod3.a")).isFalse();

    ClassSymbolImpl d = new ClassSymbolImpl("c", "mod1.c");
    d.addSuperClass(aOrB);
    d.setHasSuperClassWithoutSymbol();
    assertThat(d.canBeOrExtend("mod1.a")).isTrue();
    assertThat(d.canBeOrExtend("mod2.a")).isTrue();
    assertThat(d.canBeOrExtend("mod3.a")).isTrue();

    assertThat(new ClassSymbolImpl("foo", "foo").canBeOrExtend("object")).isTrue();
    assertThat(a.canBeOrExtend("object")).isTrue();
  }

  @Test
  void removeUsages() {
    FileInput fileInput = parse(
      "class Base: ...",
      "class A(Base):",
      "  def meth(): ..."
    );

    ClassDef classDef = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Tree.Kind.CLASSDEF));
    ClassSymbol symbol = (ClassSymbol) classDef.name().symbol();
    ((SymbolImpl) symbol).removeUsages();
    assertThat(symbol.usages()).isEmpty();
    assertThat(symbol.declaredMembers()).allMatch(member -> member.usages().isEmpty());
    assertThat(symbol.superClasses()).allMatch(superClass -> superClass.usages().isEmpty());
  }

  @Test
  void from_protobuf() throws TextFormat.ParseException {
    String protobuf = """
      name: "A"
      fully_qualified_name: "mod.A"
      super_classes: "builtins.object"
      has_decorators: true
      has_metaclass: true
      metaclass_name: "abc.ABCMeta"
      """;
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(classSymbol(protobuf), "mod");
    assertThat(classSymbol.name()).isEqualTo("A");
    assertThat(classSymbol.fullyQualifiedName()).isEqualTo("mod.A");
    assertThat(classSymbol.superClasses()).extracting(Symbol::fullyQualifiedName).containsExactly("object");
    assertThat(classSymbol.hasDecorators()).isTrue();
    assertThat(classSymbol.hasMetaClass()).isTrue();
    assertThat(classSymbol.metaclassFQN()).isEqualTo("abc.ABCMeta");
  }

  @Test
  void from_protobuf_instance_method() throws TextFormat.ParseException {
    String protobuf = """
      name: "A"
      fully_qualified_name: "mod.A"
      super_classes: "builtins.object"
      methods {
        name: "foo"
        fully_qualified_name: "mod.A.foo"
        parameters {
          name: "self"
          kind: POSITIONAL_OR_KEYWORD
        }
        has_decorators: true
      }
      """;
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(classSymbol(protobuf), "mod");
    FunctionSymbol foo = (FunctionSymbol) classSymbol.declaredMembers().iterator().next();
    assertThat(foo.isInstanceMethod()).isTrue();
  }

  @Test
  void from_protobuf_class_method() throws TextFormat.ParseException {
    String protobuf = """
      name: "A"
      fully_qualified_name: "mod.A"
      super_classes: "builtins.object"
      methods {
        name: "foo"
        fully_qualified_name: "mod.A.foo"
        parameters {
          name: "cls"
          kind: POSITIONAL_OR_KEYWORD
        }
        has_decorators: true
        is_class_method: true
      }
      """;
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(classSymbol(protobuf), "mod");
    FunctionSymbol foo = (FunctionSymbol) classSymbol.declaredMembers().iterator().next();
    assertThat(foo.isInstanceMethod()).isFalse();
  }

  @Test
  void from_protobuf_static_method() throws TextFormat.ParseException {
    String protobuf = """
      name: "A"
      fully_qualified_name: "mod.A"
      super_classes: "builtins.object"
      methods {
        name: "foo"
        fully_qualified_name: "mod.A.foo"
        parameters {
          name: "x"
          kind: POSITIONAL_OR_KEYWORD
        }
        has_decorators: true
        is_static: true
      }
      """;
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(classSymbol(protobuf), "mod");
    FunctionSymbol foo = (FunctionSymbol) classSymbol.declaredMembers().iterator().next();
    assertThat(foo.isInstanceMethod()).isFalse();
  }

  @Test
  void overloaded_methods() throws TextFormat.ParseException {
    String protobuf = """
      name: "A"
      fully_qualified_name: "mod.A"
      super_classes: "builtins.object"
      methods {
        name: "foo"
        fully_qualified_name: "mod.A.foo"
        valid_for: "39"
      }
      overloaded_methods {
        name: "foo"
        fullname: "mod.A.foo"
        valid_for: "310"
        definitions {
          name: "foo"
          fully_qualified_name: "mod.A.foo"
          has_decorators: true
        }
        definitions {
          name: "foo"
          fully_qualified_name: "mod.A.foo"
        }
      }
      
      """;
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(classSymbol(protobuf), "mod");
    Symbol foo = classSymbol.resolveMember("foo").get();
    assertThat(foo.is(Symbol.Kind.AMBIGUOUS)).isTrue();
    assertThat(((SymbolImpl) foo).validForPythonVersions()).containsExactlyInAnyOrder("39", "310");
  }

  private static SymbolsProtos.ClassSymbol classSymbol(String protobuf) throws TextFormat.ParseException {
    SymbolsProtos.ClassSymbol.Builder builder = SymbolsProtos.ClassSymbol.newBuilder();
    TextFormat.merge(protobuf, builder);
    return builder.build();
  }
}
