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

import java.util.Collections;
import java.util.HashSet;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.types.TypeShed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.python.PythonTestUtils.parse;

public class ClassSymbolImplTest {

  @Test
  public void hasUnresolvedTypeHierarchy() {
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
  public void cycle_between_super_classes() {
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
  public void should_throw_when_adding_super_class_after_super_classes_were_read() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    ClassSymbolImpl b = new ClassSymbolImpl("b", null);
    ClassSymbolImpl c = new ClassSymbolImpl("c", null);
    a.addSuperClass(b);
    assertThat(a.superClasses()).containsExactly(b);
    assertThatThrownBy(() -> a.addSuperClass(c)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  public void should_throw_when_adding_super_class_after_checking_hierarchy() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    ClassSymbolImpl b = new ClassSymbolImpl("b", null);
    ClassSymbolImpl c = new ClassSymbolImpl("c", null);
    a.addSuperClass(b);
    assertThat(a.hasUnresolvedTypeHierarchy()).isFalse();
    assertThatThrownBy(() -> a.addSuperClass(c)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  public void resolveMember() {
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
  public void resolve_inherited_member() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    ClassSymbolImpl b = new ClassSymbolImpl("b", null);
    Symbol fooB = new SymbolImpl("foo", "b.foo");
    b.addMembers(Collections.singleton(fooB));
    a.addSuperClass(b);
    assertThat(a.resolveMember("foo")).contains(fooB);
  }

  @Test
  public void resolve_overridden_member() {
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
  public void should_throw_when_adding_member_after_call_to_resolveMember() {
    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    a.addMembers(Collections.singleton(new SymbolImpl("m1", null)));
    assertThat(a.resolveMember("m1")).isNotEmpty();
    assertThatThrownBy(() -> a.addMembers(Collections.singleton(new SymbolImpl("m2", null)))).isInstanceOf(IllegalStateException.class);
  }

  @Test
  public void isOrExtends() {
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
  public void isOrExtends_non_class_symbol() {
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
  public void canBeOrExtend() {
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
  public void removeUsages() {
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
}
