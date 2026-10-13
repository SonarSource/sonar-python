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
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.DecoratorInfo;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.index.AliasDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.DecoratorDescriptor;
import org.sonar.python.index.DecoratorDescriptor.TypeReference;
import org.sonar.python.index.DescriptorsToProtobuf;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.semantic.v2.SymbolV2Impl;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.python.types.v2.TypesTestUtils;
import org.sonar.python.types.v2.matchers.HasKnownClassDecoratorEffectsPredicate;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

import static org.assertj.core.api.Assertions.assertThat;

class DecoratorDescriptorConverterTest {
  private final ProjectLevelTypeTable table = TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
  private final ConversionContext context = new ConversionContext("mod", table.lazyTypesContext(),
    (ctx, descriptor) -> PythonType.UNKNOWN, TypeOrigin.STUB);

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void factory_instance_unions_survive_indexing_and_caching(boolean cacheRoundTrip) {
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.addModule(PythonTestUtils.parseWithoutSymbols("""
      class A:
          def __call__(self, cls): return cls
      class B:
          def __call__(self, cls): return cls
      def make() -> A | B:
          return A() if condition else B()
      @make()
      class C: pass
      """), "", PythonTestUtils.pythonFile("marked.py"));
    var importedSymbols = cacheRoundTrip ? roundTrip(symbols) : symbols;
    var descriptor = (ClassDescriptor) importedSymbols.descriptorsForModule("marked").stream()
      .filter(candidate -> candidate.name().equals("C")).findFirst().orElseThrow();
    assertThat(descriptor.decorators()).containsExactly(new DecoratorDescriptor(
      List.of(new TypeReference("marked.A", true), new TypeReference("marked.B", true)),
      Optional.of(List.of(new TypeReference("marked.make", false)))));
    var importedTable = new ProjectLevelTypeTable(importedSymbols);
    var decorator = importClass(importedTable, "C").decorators().get(0);
    assertThat(decorator.calleeType().orElseThrow().type()).isSameAs(importedTable.getType("marked.make"));
    assertThat(decorator.expressionType().type()).isInstanceOfSatisfying(UnionType.class, union -> {
      assertThat(union.candidates()).allSatisfy(candidate -> assertThat(candidate).isInstanceOf(ObjectType.class));
      assertThat(union.candidates().stream().map(candidate -> ((ObjectType) candidate).type()))
        .containsExactlyInAnyOrder(importedTable.getType("marked.A"), importedTable.getType("marked.B"));
    });
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void indexed_imports_aliases_and_factory_callees_resolve_lazily(boolean cacheRoundTrip) {
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.addModule(PythonTestUtils.parseWithoutSymbols("""
      import typing
      from typing import final as alias
      from dataclasses import dataclass as factory
      @typing.final
      @alias
      class C: pass
      @(factory())
      class D: pass
      """), "", PythonTestUtils.pythonFile("marked.py"));
    var importedSymbols = cacheRoundTrip ? roundTrip(symbols) : symbols;
    var importedTable = new ProjectLevelTypeTable(importedSymbols);
    var imported = importClass(importedTable, "C");
    assertThat(imported.decorators()).hasSize(2).allSatisfy(decorator -> {
      assertThat(decorator.calleeType()).isEmpty();
      assertThat(decorator.expressionType()).isInstanceOfSatisfying(LazyTypeWrapper.class,
        lazy -> assertThat(lazy.isResolved()).isFalse());
    });
    assertThat(imported.decorators()).allSatisfy(decorator ->
      assertThat(decorator.expressionType().type()).isSameAs(importedTable.getType("typing.final")));
    assertThat(new HasKnownClassDecoratorEffectsPredicate().check(imported, TypePredicateContext.of(importedTable))).isEqualTo(TriBool.TRUE);
    var factory = importClass(importedTable, "D").decorators().get(0);
    assertThat(factory.calleeType().orElseThrow().type()).isSameAs(importedTable.getType("dataclasses.dataclass"));
    assertThat(factory.expressionType().type()).isEqualTo(PythonType.UNKNOWN);
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void missing_imported_decorator_remains_unknown(boolean cacheRoundTrip) {
    var symbols = ProjectLevelSymbolTable.empty();
    // Shadow bundled typing stubs so that final cannot resolve to a valid definition.
    symbols.insertEntry("typing", Set.of());
    symbols.addModule(PythonTestUtils.parseWithoutSymbols("""
      from typing import final
      @final
      class C:
          pass
      """), "", PythonTestUtils.pythonFile("marked.py"));
    var importedSymbols = cacheRoundTrip ? roundTrip(symbols) : symbols;
    var descriptor = (ClassDescriptor) importedSymbols.descriptorsForModule("marked").stream()
      .filter(candidate -> candidate.name().equals("C")).findFirst().orElseThrow();
    assertThat(descriptor.decorators()).containsExactly(new DecoratorDescriptor(
      List.of(new TypeReference("typing.final", false)), Optional.empty()));
    var importedTable = new ProjectLevelTypeTable(importedSymbols);
    assertThat(importedTable.getType("typing.final")).isInstanceOf(UnknownType.class);
    var imported = importClass(importedTable, "C");
    assertThat(imported.decorators().get(0).expressionType().type()).isInstanceOf(UnknownType.class);
    assertThat(new HasKnownClassDecoratorEffectsPredicate().check(imported, TypePredicateContext.of(importedTable))).isEqualTo(TriBool.FALSE);
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void indexed_anonymous_unknowns_and_calls_remain_untrusted(boolean cacheRoundTrip) {
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.insertEntry("typing", Set.of());
    symbols.addModule(PythonTestUtils.parseWithoutSymbols("""
      from typing import final
      @(unknown + other)
      @unknown_factory()
      @final()
      class D: pass
      """), "", PythonTestUtils.pythonFile("marked.py"));
    var importedSymbols = cacheRoundTrip ? roundTrip(symbols) : symbols;
    var importedTable = new ProjectLevelTypeTable(importedSymbols);
    var unknowns = importClass(importedTable, "D").decorators();
    assertThat(unknowns.get(0).expressionType().type()).isEqualTo(PythonType.UNKNOWN);
    assertThat(unknowns.get(0).calleeType()).isEmpty();
    assertThat(unknowns.get(1).calleeType().orElseThrow().type()).isEqualTo(PythonType.UNKNOWN);
    assertThat(unknowns.get(2).calleeType().orElseThrow().type()).isInstanceOf(UnknownType.class);
  }

  private static ProjectLevelSymbolTable roundTrip(ProjectLevelSymbolTable symbols) {
    var restored = ProjectLevelSymbolTable.empty();
    symbols.globalDescriptorsByModuleName().forEach((module, descriptors) -> restored.insertEntry(module,
      DescriptorsToProtobuf.fromProtobuf(DescriptorsToProtobuf.toProtobufModuleDescriptor(descriptors))));
    return restored;
  }

  private static ClassType importClass(ProjectLevelTypeTable table, String name) {
    var source = TypesTestUtils.parseAndInferTypes(table, PythonTestUtils.pythonFile("usage.py"),
      "from marked import " + name + " as Imported\nImported");
    Name imported = PythonTestUtils.getLastDescendant(source, tree -> tree.is(Tree.Kind.NAME));
    return (ClassType) imported.typeV2();
  }

  @Test
  void resolved_values_and_unresolved_imports_serialize_as_the_same_named_reference() {
    var resolved = DecoratorDescriptorConverter.toDescriptor(new DecoratorInfo(TypeWrapper.of(table.getType("typing.final"))));
    var unresolved = DecoratorDescriptorConverter.toDescriptor(new DecoratorInfo(TypeWrapper.of(new UnknownType.UnresolvedImportType("typing.final"))));
    assertThat(unresolved).isEqualTo(resolved);
    var descriptor = new ClassDescriptor.ClassDescriptorBuilder().withName("C").withFullyQualifiedName("mod.C")
      .withDecorators(List.of(unresolved)).build();
    var restored = (ClassDescriptor) DescriptorsToProtobuf.fromProtobuf(DescriptorsToProtobuf.toProtobufModuleDescriptor(Set.of(descriptor)))
      .iterator().next();
    assertThat(restored.decorators()).containsExactly(unresolved);
    assertThat(DecoratorDescriptorConverter.toType(context, restored.decorators().get(0)).expressionType().type())
      .isSameAs(table.getType("typing.final"));
  }

  @Test
  void known_limitation_imported_decorator_resolves_to_redefinition_after_cache_round_trip() {
    var source = PythonTestUtils.parseWithoutSymbols("""
      def marker(cls): return cls
      @marker
      class C: pass
      def marker(cls): return object
      """);
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.addModule(source, "", PythonTestUtils.pythonFile("mod.py"));
    ClassDef definition = PythonTestUtils.getLastDescendant(source, tree -> tree.is(Tree.Kind.CLASSDEF));
    List<FunctionDef> functions = PythonTestUtils.getAllDescendant(source, tree -> tree.is(Tree.Kind.FUNCDEF));
    var sourceDecorator = ((ClassType) definition.name().typeV2()).decorators().get(0).expressionType().type();
    assertThat(sourceDecorator).isSameAs(functions.get(0).name().typeV2());
    assertThat(sourceDecorator.definitionLocation().orElseThrow().startLine()).isEqualTo(1);

    var cached = DescriptorsToProtobuf.toProtobufModuleDescriptor(symbols.descriptorsForModule("mod"));
    var restoredSymbols = ProjectLevelSymbolTable.empty();
    restoredSymbols.insertEntry("mod", DescriptorsToProtobuf.fromProtobuf(cached));
    var importedTable = new ProjectLevelTypeTable(restoredSymbols);
    var imported = (ClassType) importedTable.getType("mod.C");
    var importedDecorator = imported.decorators().get(0).expressionType().type();
    // Accepted lookup of the later declaration: CLP-1206
    assertThat(importedDecorator).isSameAs(importedTable.getType("mod.marker"));
    assertThat(importedDecorator.definitionLocation().orElseThrow().startLine()).isEqualTo(4);
  }

  @Test
  void known_limitation_cache_round_trip_can_certify_a_different_callable_with_a_marker_fqn() {
    var impostor = new FunctionTypeBuilder("final").withFullyQualifiedName("typing.final").build();
    var source = new ClassTypeBuilder("C", "mod.C")
      .withDecorators(List.of(new DecoratorInfo(TypeWrapper.of(impostor)))).build();
    var predicate = new HasKnownClassDecoratorEffectsPredicate();
    var predicateContext = TypePredicateContext.of(table);
    assertThat(predicate.check(source, predicateContext)).isEqualTo(TriBool.FALSE);
    var descriptor = new PythonTypeToDescriptorConverter().convert("mod", new SymbolV2Impl("C"), Set.of(source));
    var cached = DescriptorsToProtobuf.toProtobufModuleDescriptor(Set.of(descriptor));
    var restored = (ClassDescriptor) DescriptorsToProtobuf.fromProtobuf(cached).iterator().next();
    var imported = (ClassType) new ClassDescriptorToPythonTypeConverter().convert(context, restored);
    // Accepted loss of callable identity during conversion: CLP-1206
    assertThat(imported.decorators().get(0).expressionType().type()).isSameAs(table.getType("typing.final"));
    assertThat(predicate.check(imported, predicateContext)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void source_decorators_survive_cache_and_import_with_unions_factories_and_unknowns() {
    ClassDef definition = TypesTestUtils.lastClassDef("""
      from typing import final
      from typing_extensions import final as other
      from dataclasses import dataclass
      @(final if condition else other)
      @missing
      @dataclass(slots=True)
      @unknown_factory()
      class C: pass
      """);
    var sourceType = (ClassType) definition.name().typeV2();
    var descriptor = (ClassDescriptor) new PythonTypeToDescriptorConverter()
      .convert("mod", new SymbolV2Impl("C"), Set.of(sourceType));
    var proto = DescriptorsToProtobuf.toProtobufModuleDescriptor(Set.of(descriptor));
    assertThat(proto.getDescriptors().getClassDescriptors(0).getDecoratorsCount()).isEqualTo(4);
    var restored = (ClassDescriptor) DescriptorsToProtobuf.fromProtobuf(proto).iterator().next();
    assertThat(restored.decorators()).isEqualTo(descriptor.decorators());
    var imported = (ClassType) new ClassDescriptorToPythonTypeConverter().convert(context, restored);
    assertThat(imported.decorators()).hasSize(4);
    var union = imported.decorators().get(0).expressionType().type();
    assertThat(union).isInstanceOfSatisfying(UnionType.class,
      alternatives -> assertThat(alternatives.candidates().stream().map(FullyQualifiedNameHelper::getFullyQualifiedName))
        .containsExactlyInAnyOrder(Optional.of("typing.final"), Optional.of("typing_extensions.final")));
    assertThat(imported.decorators().get(1).expressionType().type()).isEqualTo(PythonType.UNKNOWN);
    var factory = imported.decorators().get(2);
    assertThat(factory.calleeType().orElseThrow().type()).isSameAs(table.getType("dataclasses.dataclass"));
    assertThat(factory.expressionType().type()).isInstanceOf(UnionType.class);
    assertThat(imported.decorators().get(3).calleeType()).isPresent();
    assertThat(imported.decorators().get(3).calleeType().orElseThrow().type()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void named_references_remain_lazy_and_resolve_through_reexports() {
    var original = new FunctionDescriptor.FunctionDescriptorBuilder().withName("marker").withFullyQualifiedName("lib.marker").build();
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.insertEntry("lib", Set.of(original));
    symbols.insertEntry("wrapper", Set.of(new AliasDescriptor("alias", "wrapper.alias", original)));
    var project = new ProjectLevelTypeTable(symbols);
    var ctx = new ConversionContext("mod", project.lazyTypesContext(), (c, d) -> PythonType.UNKNOWN, TypeOrigin.STUB);
    var descriptor = new DecoratorDescriptor(List.of(new TypeReference("wrapper.alias", false)), Optional.empty());
    var info = DecoratorDescriptorConverter.toType(ctx, descriptor);
    assertThat(info.expressionType()).isInstanceOfSatisfying(LazyTypeWrapper.class,
      lazy -> assertThat(lazy.isResolved()).isFalse());
    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(info.expressionType().type())).contains("lib.marker");
    assertThat(DecoratorDescriptorConverter.toDescriptor(info).expressionTypes()).containsExactly(new TypeReference("lib.marker", false));
  }

  @Test
  void callable_instances_remain_instances_after_import() {
    var info = new DecoratorInfo(TypeWrapper.of(ObjectType.fromType(TypesTestUtils.STR_TYPE)));
    var descriptor = DecoratorDescriptorConverter.toDescriptor(info);
    assertThat(descriptor.expressionTypes()).containsExactly(new TypeReference(
      FullyQualifiedNameHelper.getFullyQualifiedName(TypesTestUtils.STR_TYPE).orElseThrow(), true));
    var restored = DecoratorDescriptorConverter.toType(context, descriptor);
    assertThat(restored.expressionType().type()).isInstanceOfSatisfying(ObjectType.class,
      instance -> assertThat(instance.type()).isEqualTo(TypesTestUtils.STR_TYPE));
  }

  @Test
  void unknown_alternatives_are_explicit_in_descriptors_and_conservative_in_types() {
    var decorator = new DecoratorDescriptor(List.of(new TypeReference("typing.final", false), TypeReference.UNKNOWN), Optional.empty());
    var descriptor = new ClassDescriptor.ClassDescriptorBuilder().withName("C").withFullyQualifiedName("mod.C")
      .withDecorators(List.of(decorator)).build();
    var restored = (ClassDescriptor) DescriptorsToProtobuf.fromProtobuf(DescriptorsToProtobuf.toProtobufModuleDescriptor(Set.of(descriptor)))
      .iterator().next();
    assertThat(restored.decorators()).containsExactly(decorator);
    assertThat(DecoratorDescriptorConverter.toType(context, decorator).expressionType().type()).isEqualTo(PythonType.UNKNOWN);
  }
}
