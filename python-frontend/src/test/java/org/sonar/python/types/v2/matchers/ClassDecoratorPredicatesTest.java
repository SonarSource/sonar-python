/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.types.v2.matchers;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.DecoratorInfo;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.index.AliasDescriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.semantic.v2.TestProject;
import org.sonar.python.semantic.v2.typetable.TypeTable;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ClassDecoratorPredicatesTest {
  private final TypePredicate direct = new HasKnownClassDecoratorEffectsPredicate();
  private final TypePredicate hierarchy = new HasKnownClassDecoratorEffectsInHierarchyPredicate();
  private final TypePredicateContext ctx = TypePredicateContext.of(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE);

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void self_types_preserve_direct_and_inherited_decorator_effects(boolean instance) {
    var custom = decorated(PythonType.UNKNOWN);
    var child = new ClassTypeBuilder("Child", "mod.Child").withSuperClasses(custom).build();
    var marked = decorated(ctx.typeTable().getType("typing.final"));
    var unresolved = new ClassTypeBuilder("Unresolved", "mod.Unresolved").withSuperClasses(PythonType.UNKNOWN).build();
    PythonType customSelf = SelfType.of(custom);
    PythonType childSelf = SelfType.of(child);
    PythonType markedSelf = SelfType.of(marked);
    PythonType unresolvedSelf = SelfType.of(unresolved);
    if (instance) {
      customSelf = ObjectType.fromType(customSelf);
      childSelf = ObjectType.fromType(childSelf);
      markedSelf = ObjectType.fromType(markedSelf);
      unresolvedSelf = ObjectType.fromType(unresolvedSelf);
    }
    assertThat(direct.check(customSelf, ctx)).isEqualTo(TriBool.FALSE);
    assertThat(hierarchy.check(customSelf, ctx)).isEqualTo(TriBool.FALSE);
    assertThat(direct.check(childSelf, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(hierarchy.check(childSelf, ctx)).isEqualTo(TriBool.FALSE);
    assertThat(direct.check(markedSelf, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(hierarchy.check(markedSelf, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(hierarchy.check(unresolvedSelf, ctx)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void hierarchy_matcher_unwraps_instances_and_does_not_establish_member_absence() {
    var project = new TestProject();
    var expression = project.lastExpression("from typing import final\n@final\nclass C: pass\nC()");
    var context = mock(SubscriptionContext.class);
    when(context.typeTable()).thenReturn(project.projectLevelTypeTable());
    var matcher = InternalTypeMatchers.hasKnownClassDecoratorEffectsInHierarchy();
    assertThat(matcher.evaluateFor(expression, context)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.isObjectSatisfying(matcher).evaluateFor(expression, context)).isEqualTo(TriBool.TRUE);
    // Accepted limitation: the member API still treats all decorated source classes conservatively.
    assertThat(TypeMatchers.hasMember("__getitem__").evaluateFor(expression, context)).isEqualTo(TriBool.UNKNOWN);
    assertThat(direct.check(PythonType.UNKNOWN, ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(hierarchy.check(PythonType.UNKNOWN, ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(direct.check(ctx.typeTable().getType("typing.final"), ctx)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void safety_matcher_handles_non_applicable_values_and_expression_unions() {
    var matcher = InternalTypeMatchers.hasKnownClassDecoratorEffectsInHierarchy();
    var context = mock(SubscriptionContext.class);
    when(context.typeTable()).thenReturn(ctx.typeTable());
    var expression = mock(Expression.class);
    var function = ctx.typeTable().getType("typing.final");
    var safe = decorated(function);
    var unsafe = decorated(PythonType.UNKNOWN);
    for (var type : List.of(function, safe, ObjectType.fromType(safe),
      UnionType.or(function, ObjectType.fromType(safe)))) {
      when(expression.typeV2()).thenReturn(type);
      assertThat(matcher.evaluateFor(expression, context)).isEqualTo(TriBool.TRUE);
    }
    for (var type : List.of(unsafe, ObjectType.fromType(unsafe),
      UnionType.or(function, unsafe), UnionType.or(safe, ObjectType.fromType(unsafe)))) {
      when(expression.typeV2()).thenReturn(type);
      assertThat(matcher.evaluateFor(expression, context)).isEqualTo(TriBool.FALSE);
    }
    for (var type : List.of(PythonType.UNKNOWN, ObjectType.fromType(PythonType.UNKNOWN),
      UnionType.or(safe, PythonType.UNKNOWN))) {
      when(expression.typeV2()).thenReturn(type);
      assertThat(matcher.evaluateFor(expression, context)).isEqualTo(TriBool.UNKNOWN);
    }
  }

  @Test
  void imported_custom_base_effects_prevent_a_definitive_hierarchy_guard() {
    var project = new TestProject();
    project.addModule("base.py", """
      def add_protocol(cls):
          cls.__getitem__ = lambda self, index: 42
          return cls
      @add_protocol
      class Base: pass
      """);
    var expression = project.lastExpression("""
      from base import Base
      from typing import final
      class PlainChild(Base): pass
      @final
      class MarkedChild(PlainChild): pass
      class OtherBase: pass
      class Child(OtherBase, MarkedChild): pass
      Child
      """);
    var context = mock(SubscriptionContext.class);
    when(context.typeTable()).thenReturn(project.projectLevelTypeTable());
    assertThat(direct.check(expression.typeV2(), TypePredicateContext.of(project.projectLevelTypeTable()))).isEqualTo(TriBool.TRUE);
    assertThat(InternalTypeMatchers.hasKnownClassDecoratorEffectsInHierarchy().evaluateFor(expression, context)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void unresolved_bases_are_unknown_but_unaccounted_for_decorators_are_false() {
    var plain = new ClassType("Plain", "mod.Plain");
    var legacy = new ClassTypeBuilder("Legacy", "mod.Legacy").withHasDecorators(true).build();
    assertThat(direct.check(legacy, ctx)).isEqualTo(TriBool.FALSE);
    for (var base : List.of(PythonType.UNKNOWN, new UnknownType.UnresolvedImportType("mod.Base"))) {
      var child = new ClassTypeBuilder("Child", "mod.Child").withSuperClasses(base).build();
      assertThat(direct.check(child, ctx)).isEqualTo(TriBool.TRUE);
      assertThat(hierarchy.check(child, ctx)).isEqualTo(TriBool.UNKNOWN);
    }
    for (var bases : List.of(List.of(PythonType.UNKNOWN, legacy), List.of(legacy, PythonType.UNKNOWN),
      List.of(UnionType.or(plain, legacy)))) {
      var child = new ClassTypeBuilder("Child", "mod.Child").withSuperClasses(bases.toArray(PythonType[]::new)).build();
      assertThat(hierarchy.check(child, ctx)).isEqualTo(TriBool.FALSE);
    }
    var child = new ClassTypeBuilder("Child", "mod.Child").withSuperClasses(UnionType.or(plain,
      new ClassType("Other", "mod.Other"))).build();
    assertThat(hierarchy.check(child, ctx)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void cyclic_hierarchies_terminate_without_claiming_inheritance_validity() {
    var cyclic = new ClassType("Cyclic", "mod.Cyclic");
    cyclic.superClasses().add(TypeWrapper.of(cyclic));
    assertThat(hierarchy.check(cyclic, ctx)).isEqualTo(TriBool.TRUE);
    cyclic.superClasses().add(TypeWrapper.UNKNOWN_TYPE_WRAPPER);
    assertThat(hierarchy.check(cyclic, ctx)).isEqualTo(TriBool.UNKNOWN);
  }

  @ParameterizedTest
  @ValueSource(strings = {"typing.runtime_checkable", "typing_extensions.runtime_checkable"})
  void runtime_checkable_has_known_direct_and_inherited_operand_and_item_effects(String marker) {
    var protocol = decorated(ctx.typeTable().getType(marker));
    var child = new ClassTypeBuilder("Child", "mod.Child").withSuperClasses(protocol).build();
    assertThat(direct.check(protocol, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(hierarchy.check(protocol, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(hierarchy.check(child, ctx)).isEqualTo(TriBool.TRUE);
    var mixed = new ClassTypeBuilder("Mixed", "mod.Mixed").withDecorators(List.of(
      new DecoratorInfo(TypeWrapper.of(ctx.typeTable().getType(marker))),
      new DecoratorInfo(TypeWrapper.UNKNOWN_TYPE_WRAPPER))).build();
    var mixedChild = new ClassTypeBuilder("Child", "mod.Child").withSuperClasses(mixed).build();
    assertThat(hierarchy.check(mixedChild, ctx)).isEqualTo(TriBool.FALSE);

    var matcher = InternalTypeMatchers.hasKnownClassDecoratorEffectsInHierarchy();
    var context = mock(SubscriptionContext.class);
    when(context.typeTable()).thenReturn(ctx.typeTable());
    var expression = mock(Expression.class);
    for (var type : List.of(protocol, child)) {
      when(expression.typeV2()).thenReturn(ObjectType.fromType(type));
      assertThat(matcher.evaluateFor(expression, context)).isEqualTo(TriBool.TRUE);
    }
    when(expression.typeV2()).thenReturn(protocol);
    assertThat(matcher.evaluateFor(expression, context)).isEqualTo(TriBool.TRUE);
    when(expression.typeV2()).thenReturn(PythonType.UNKNOWN);
    assertThat(matcher.evaluateFor(expression, context)).isEqualTo(TriBool.UNKNOWN);
    for (var type : List.of(ObjectType.fromType(mixed), ObjectType.fromType(mixedChild),
      UnionType.or(ObjectType.fromType(protocol), ObjectType.fromType(mixed)))) {
      when(expression.typeV2()).thenReturn(type);
      assertThat(matcher.evaluateFor(expression, context)).isEqualTo(TriBool.FALSE);
    }
  }

  @Test
  void type_check_only_has_known_effects_only_without_a_source_definition() {
    var marker = ctx.typeTable().getType("typing.type_check_only");
    var decorators = List.of(new DecoratorInfo(TypeWrapper.of(marker)));
    var stub = new ClassTypeBuilder("Stub", "mod.Stub").withDecorators(decorators).build();
    var source = new ClassTypeBuilder("Source", "mod.Source").withDecorators(decorators)
      .withDefinitionLocation(mock(LocationInFile.class)).build();
    var impostor = new FunctionTypeBuilder("type_check_only").withFullyQualifiedName("typing.type_check_only").build();
    var child = new ClassTypeBuilder("Child", "mod.Child").withSuperClasses(stub).build();
    assertThat(direct.check(stub, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(hierarchy.check(child, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(direct.check(source, ctx)).isEqualTo(TriBool.FALSE);
    assertThat(direct.check(decorated(impostor), ctx)).isEqualTo(TriBool.FALSE);
    var mixed = new ClassTypeBuilder("Mixed", "mod.Mixed").withDecorators(List.of(
      new DecoratorInfo(TypeWrapper.of(marker)), new DecoratorInfo(TypeWrapper.UNKNOWN_TYPE_WRAPPER))).build();
    assertThat(direct.check(mixed, ctx)).isEqualTo(TriBool.FALSE);
    var called = new ClassTypeBuilder("Called", "mod.Called").withDecorators(List.of(
      new DecoratorInfo(TypeWrapper.of(marker), TypeWrapper.of(marker)))).build();
    assertThat(direct.check(called, ctx)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void marker_identity_uses_canonical_types_even_when_the_definition_is_relocated() {
    var project = new TestProject();
    var definition = new FunctionDescriptor.FunctionDescriptorBuilder().withName("marker").withFullyQualifiedName("typing_extensions.marker").build();
    var stubs = project.projectLevelSymbolTable().typeShedDescriptorsProvider();
    stubs.descriptorsForModule("typing_extensions").put("marker", definition);
    stubs.descriptorsForModule("typing").put("disjoint_base", new AliasDescriptor("disjoint_base", "typing.disjoint_base", definition));
    var context = TypePredicateContext.of(project.projectLevelTypeTable());
    var canonical = context.typeTable().getType("typing.disjoint_base");
    assertThat(direct.check(decorated(canonical), context)).isEqualTo(TriBool.TRUE);
    // Matching text does not establish identity with a different callable.
    var impostor = new FunctionTypeBuilder("disjoint_base").withFullyQualifiedName("typing.disjoint_base").build();
    assertThat(direct.check(decorated(impostor), context)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void overloaded_markers_match_their_canonical_callable_alternatives() {
    var first = new FunctionTypeBuilder("final").withFullyQualifiedName("typing.final").withTypeOrigin(TypeOrigin.STUB).build();
    var second = new FunctionTypeBuilder("final").withFullyQualifiedName("typing.final").withTypeOrigin(TypeOrigin.STUB).build();
    var table = mock(TypeTable.class);
    when(table.getType(anyString())).thenReturn(PythonType.UNKNOWN);
    when(table.getType("typing.final")).thenReturn(UnionType.or(first, second));
    var context = TypePredicateContext.of(table);
    assertThat(direct.check(decorated(UnionType.or(first, second)), context)).isEqualTo(TriBool.TRUE);
    assertThat(direct.check(decorated(first), context)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void decorator_and_expression_unions_require_every_alternative_to_be_known() {
    var marker = ctx.typeTable().getType("typing.final");
    var other = ctx.typeTable().getType("typing_extensions.final");
    assertThat(direct.check(decorated(UnionType.or(marker, other)), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(direct.check(decorated(UnionType.or(marker, PythonType.UNKNOWN)), ctx)).isEqualTo(TriBool.FALSE);
    var custom = decorated(PythonType.UNKNOWN);
    assertThat(TypePredicateUtils.evaluate(hierarchy, UnionType.or(decorated(marker), custom), ctx)).isEqualTo(TriBool.FALSE);
    assertThat(TypePredicateUtils.evaluate(hierarchy, UnionType.or(decorated(marker), new ClassType("Plain", "mod.Plain")), ctx))
      .isEqualTo(TriBool.TRUE);
    var called = new ClassTypeBuilder("Called", "mod.Called").withDecorators(List.of(
      new DecoratorInfo(TypeWrapper.of(marker), TypeWrapper.of(marker)))).build();
    assertThat(direct.check(called, ctx)).isEqualTo(TriBool.FALSE);
  }

  private static ClassType decorated(PythonType decorator) {
    return new ClassTypeBuilder("C", "mod.C").withDecorators(List.of(new DecoratorInfo(TypeWrapper.of(decorator)))).build();
  }
}
