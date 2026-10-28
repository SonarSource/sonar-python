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
package org.sonar.python.types.v2;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.DecoratorInfo;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.v2.FunctionType;
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
import org.sonar.python.index.DecoratorProtobuf;
import org.sonar.python.index.DescriptorsToProtobuf;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.SymbolV2Impl;
import org.sonar.python.semantic.v2.converter.ClassDescriptorToPythonTypeConverter;
import org.sonar.python.semantic.v2.converter.ConversionContext;
import org.sonar.python.semantic.v2.converter.DecoratorDescriptorConverter;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.typeshed.ClassSymbolToDescriptorConverter;
import org.sonar.python.semantic.v2.typeshed.FunctionSymbolToDescriptorConverter;
import org.sonar.python.semantic.v2.typeshed.OverloadedFunctionSymbolToDescriptorConverter;
import org.sonar.python.semantic.v2.typeshed.VarSymbolToDescriptorConverter;
import org.sonar.python.types.v2.matchers.HasKnownClassDecoratorEffectsInHierarchyPredicate;
import org.sonar.python.types.v2.matchers.HasKnownClassDecoratorEffectsPredicate;
import org.sonar.python.types.v2.matchers.IsObjectSatisfyingPredicate;
import org.sonar.python.types.v2.matchers.TypePredicateContext;
import org.sonar.python.types.v2.matchers.TypePredicateUtils;
import org.sonar.python.types.protobuf.DescriptorsProtos;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;

class ResolvedClassDecoratorsTest {
  private final HasKnownClassDecoratorEffectsPredicate predicate = new HasKnownClassDecoratorEffectsPredicate();

  @ParameterizedTest
  @ValueSource(strings = {"typing.final", "typing_extensions.final", "typing.disjoint_base", "typing_extensions.disjoint_base",
    "typing.runtime_checkable", "typing_extensions.runtime_checkable"})
  void project_function_markers_are_not_recognized_after_indexing_and_caching(String marker) {
    int split = marker.lastIndexOf('.');
    String module = marker.substring(0, split);
    String name = marker.substring(split + 1);
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.addModule(PythonTestUtils.parseWithoutSymbols("def " + name + "(cls): return cls"), "", PythonTestUtils.pythonFile(module + ".py"));
    String code = "from " + module + " import " + name + " as mark\n@mark\nclass C: pass";
    symbols.addModule(PythonTestUtils.parseWithoutSymbols(code), "", PythonTestUtils.pythonFile("marked.py"));
    var cached = ProjectLevelSymbolTable.empty();
    symbols.globalDescriptorsByModuleName().forEach((moduleName, descriptors) -> cached.insertEntry(moduleName,
      DescriptorsToProtobuf.fromProtobuf(DescriptorsToProtobuf.toProtobufModuleDescriptor(descriptors))));

    for (var symbolTable : List.of(symbols, cached)) {
      var table = new ProjectLevelTypeTable(symbolTable);
      for (var decorated : List.of(type(code, table), (ClassType) table.getType("marked.C"))) {
        assertThat(decorated.decorators().get(0).expressionType().type()).isInstanceOfSatisfying(FunctionType.class,
          function -> assertThat(function.typeOrigin()).isEqualTo(TypeOrigin.LOCAL));
        assertThat(check(decorated, table)).isEqualTo(TriBool.FALSE);
        var child = new ClassTypeBuilder("Child", "usage.Child").withSuperClasses(decorated).build();
        assertThat(new HasKnownClassDecoratorEffectsInHierarchyPredicate().check(child, TypePredicateContext.of(table)))
          .isEqualTo(TriBool.FALSE);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"factory()", "Known, factory()", "factory(), Known"})
  void unresolved_bases_remain_unknown_after_indexing_and_caching(String bases) {
    String code = """
      def factory():
          return unknown
      class Known: pass
      class C(%s): pass
      """.formatted(bases);
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.addModule(PythonTestUtils.parseWithoutSymbols(code), "", PythonTestUtils.pythonFile("usage.py"));
    var cached = ProjectLevelSymbolTable.empty();
    symbols.globalDescriptorsByModuleName().forEach((moduleName, descriptors) -> cached.insertEntry(moduleName,
      DescriptorsToProtobuf.fromProtobuf(DescriptorsToProtobuf.toProtobufModuleDescriptor(descriptors))));

    var hierarchy = new HasKnownClassDecoratorEffectsInHierarchyPredicate();
    for (var symbolTable : List.of(symbols, cached)) {
      var table = new ProjectLevelTypeTable(symbolTable);
      var context = TypePredicateContext.of(table);
      var source = type(code, table);
      assertThat(hierarchy.check(source, context)).isEqualTo(TriBool.UNKNOWN);
      var imported = (ClassType) table.getType("usage.C");
      assertThat(hierarchy.check(imported, context)).isEqualTo(TriBool.UNKNOWN);
      assertThat(imported.superClasses()).anySatisfy(base -> assertThat(base.type()).isInstanceOf(UnknownType.class));
      if (bases.contains("Known")) {
        assertThat(imported.superClasses()).anySatisfy(base -> assertThat(base.type()).isEqualTo(table.getType("usage.Known")));
      }
    }
  }

  @Test
  void class_with_a_marker_name_is_not_recognized() {
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.addModule(PythonTestUtils.parseWithoutSymbols("""
      class final:
          def __new__(cls, decorated):
              decorated.__class_getitem__ = classmethod(lambda cls, key: 42)
              return decorated
      """), "", PythonTestUtils.pythonFile("typing.py"));
    var table = new ProjectLevelTypeTable(symbols);
    var type = type("from typing import final\n@final\nclass C: pass", table);
    assertThat(type.decorators().get(0).expressionType().type()).isInstanceOf(ClassType.class);
    assertThat(check(type, table)).isEqualTo(TriBool.FALSE);
  }

  @ParameterizedTest
  @ValueSource(strings = {"typing.final", "typing_extensions.final", "typing.disjoint_base", "typing_extensions.disjoint_base"})
  void direct_aliased_markers_preserve_identities(String marker) {
    int split = marker.lastIndexOf('.');
    String code = "from " + marker.substring(0, split) + " import " + marker.substring(split + 1)
      + " as mark\n@mark\nclass Marked: pass";
    var table = markerTable(marker);
    var type = type(code, table);
    assertThat(type.hasDecorators()).isTrue();
    assertThat(names(type)).containsExactly(marker);
    assertThat(check(type, table)).isEqualTo(TriBool.TRUE);
    // Decorator classification does not change member queries.
    assertThat(type.instancesHaveMember("missing")).isEqualTo(TriBool.UNKNOWN);
  }

  @ParameterizedTest
  @ValueSource(strings = {"typing.final", "typing_extensions.final", "typing.disjoint_base", "typing_extensions.disjoint_base"})
  void qualified_markers_preserve_their_canonical_names(String marker) {
    int split = marker.lastIndexOf('.');
    String code = "import " + marker.substring(0, split) + " as markers\n@markers."
      + marker.substring(split + 1) + "\nclass Marked: pass";
    var table = markerTable(marker);
    var type = type(code, table);
    assertThat(names(type)).containsExactly(marker);
    assertThat(check(type, table)).isEqualTo(TriBool.TRUE);
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "@unknown\n@mark", "@mark\n@unknown", "@mark()", "@unresolved", "@local_marker",
    "@dataclass", "@dataclass(slots=True)", "@mark\n@dataclass", "@deprecated('old')", "@type_check_only"
  })
  void calls_unresolved_and_other_decorators_stay_conservative(String decorators) {
    String code = """
      from typing import final as mark, type_check_only
      from typing_extensions import deprecated
      from dataclasses import dataclass
      def unknown(cls): return cls
      def local_marker(cls): return cls
      """ + decorators + "\nclass Marked: pass";
    assertThat(check(type(code))).isEqualTo(TriBool.FALSE);
  }

  @Test
  void partially_resolved_stacks_keep_the_unknown_entry() {
    var type = type("from typing import final\n@final\n@unresolved\nclass Marked: pass");
    assertThat(names(type)).containsExactly("typing.final", "");
    assertThat(check(type)).isEqualTo(TriBool.FALSE);
    var called = type("from typing import final\n@final()\nclass Marked: pass");
    assertThat(names(called)).containsExactly("");
  }

  @Test
  void shadowed_marker_names_are_not_typing_markers() {
    String code = "from typing import final\ndef final(cls): return cls\n@final\nclass Marked: pass";
    assertThat(check(type(code))).isEqualTo(TriBool.FALSE);
  }

  @ParameterizedTest
  @ValueSource(strings = {"custom", "my_wrapper"})
  void reassigned_marker_names_remain_conservative(String replacement) {
    String code = "from typing import final\ndef custom(cls): return cls\nfinal = " + replacement + "\n@final\nclass Marked: pass";
    var type = type(code);
    assertThat(check(type)).isEqualTo(TriBool.FALSE);
    var descriptor = new PythonTypeToDescriptorConverter().convert("usage", new SymbolV2Impl("Marked"), Set.of(type));
    var restored = DescriptorsToProtobuf.fromProtobuf(DescriptorsToProtobuf.toProtobufModuleDescriptor(Set.of(descriptor)));
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.insertEntry("usage", restored);
    var table = new ProjectLevelTypeTable(symbols);
    assertThat(check(table.getType("usage.Marked"), table)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void plain_classes_and_unknown_types_have_distinct_results() {
    assertThat(check(type("class Plain: pass"))).isEqualTo(TriBool.TRUE);
    assertThat(check(PythonType.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    var marker = type("from typing import final\n@final\nclass Marked: pass");
    assertThat(check(ObjectType.fromType(marker))).isEqualTo(TriBool.TRUE);
    assertThat(TypePredicateUtils.evaluate(new IsObjectSatisfyingPredicate(predicate), ObjectType.fromType(marker),
      TypePredicateContext.of(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE))).isEqualTo(TriBool.TRUE);
    var custom = type("@custom\nclass Custom: pass");
    assertThat(check(ObjectType.fromType(custom))).isEqualTo(TriBool.FALSE);
    assertThat(TypePredicateUtils.evaluate(new IsObjectSatisfyingPredicate(predicate), ObjectType.fromType(custom),
      TypePredicateContext.of(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE))).isEqualTo(TriBool.FALSE);
    assertThat(check(UnionType.or(marker, custom))).isEqualTo(TriBool.FALSE);
    assertThat(check(UnionType.or(marker, type("class Plain: pass")))).isEqualTo(TriBool.TRUE);
  }

  @Test
  void source_metadata_survives_export_cache_import_and_aliases() {
    var type = type("from typing import final\n@final\n@unresolved\nclass Marked: pass");
    var descriptor = new PythonTypeToDescriptorConverter().convert("usage", new SymbolV2Impl("Marked"), Set.of(type));
    var module = DescriptorsToProtobuf.toProtobufModuleDescriptor(Set.of(descriptor));
    var restored = (ClassDescriptor) DescriptorsToProtobuf.fromProtobuf(module).iterator().next();
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.insertEntry("usage", Set.of(restored));
    symbols.insertEntry("alias", Set.of(new AliasDescriptor("Alias", "alias.Alias", restored)));
    var table = new ProjectLevelTypeTable(symbols);
    for (String name : List.of("usage.Marked", "alias.Alias")) {
      var imported = (ClassType) table.getType(name);
      assertThat(imported.hasDecorators()).isTrue();
      assertThat(imported.decorators().stream().map(DecoratorDescriptorConverter::toDescriptor))
        .containsExactlyElementsOf(((ClassDescriptor) descriptor).decorators());
      assertThat(check(imported, table)).isEqualTo(TriBool.FALSE);
    }
  }

  @Test
  void old_symbols_and_cache_entries_remain_conservative() {
    var proto = SymbolsProtos.ClassSymbol.newBuilder().setName("Legacy").setFullyQualifiedName("mod.Legacy").setHasDecorators(true).build();
    var functionConverter = new FunctionSymbolToDescriptorConverter();
    var converter = new ClassSymbolToDescriptorConverter(new VarSymbolToDescriptorConverter(), functionConverter,
      new OverloadedFunctionSymbolToDescriptorConverter(functionConverter), Set.of("314"));
    var legacyDescriptor = DescriptorsProtos.ClassDescriptor.newBuilder().setName("Legacy").setFullyQualifiedName("mod.Legacy")
      .setHasDecorators(true).build();
    var legacyModule = DescriptorsProtos.ModuleDescriptor.newBuilder()
      .setDescriptors(DescriptorsProtos.DescriptorList.newBuilder().addClassDescriptors(legacyDescriptor)).build();
    var restored = (ClassDescriptor) DescriptorsToProtobuf.fromProtobuf(legacyModule).iterator().next();
    for (var descriptor : List.of(converter.convert(proto), restored)) {
      var symbols = ProjectLevelSymbolTable.empty();
      symbols.insertEntry("mod", Set.of(descriptor));
      var table = new ProjectLevelTypeTable(symbols);
      var imported = (ClassType) table.getType("mod.Legacy");
      assertThat(imported.hasDecorators()).isTrue();
      assertThat(imported.decorators()).isEmpty();
      assertThat(check(imported, table)).isEqualTo(TriBool.FALSE);
    }
  }

  @Test
  void typeshed_symbol_metadata_reaches_types() {
    var decorator = new DecoratorDescriptor(List.of(new DecoratorDescriptor.TypeReference("typing_extensions.disjoint_base", false)), Optional.empty());
    var builder = SymbolsProtos.ClassSymbol.newBuilder().setName("Marked").setFullyQualifiedName("mod.Marked").setHasDecorators(true)
      .addDecorators(DecoratorProtobuf.toProtobuf(decorator));
    var proto = builder.build();
    var functionConverter = new FunctionSymbolToDescriptorConverter();
    var converter = new ClassSymbolToDescriptorConverter(new VarSymbolToDescriptorConverter(), functionConverter,
      new OverloadedFunctionSymbolToDescriptorConverter(functionConverter), Set.of("314"));
    var descriptor = converter.convert(proto);
    var table = markerTable("typing_extensions.disjoint_base");
    var context = new ConversionContext("mod", new LazyTypesContext(table),
      (ctx, desc) -> PythonType.UNKNOWN, TypeOrigin.STUB);
    assertThat(check(new ClassDescriptorToPythonTypeConverter().convert(context, descriptor), table)).isEqualTo(TriBool.TRUE);
  }

  @ParameterizedTest
  @ValueSource(strings = {"int", "str", "list"})
  void bundled_builtin_markers_are_recognized_without_changing_legacy_member_checks(String name) {
    var type = (ClassType) TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE.getBuiltinsModule().resolveMember(name).orElseThrow();
    assertThat(names(type)).containsExactly("typing_extensions.disjoint_base");
    assertThat(check(type)).isEqualTo(TriBool.TRUE);
    // Builtins retain their existing special treatment in the legacy member API.
    assertThat(type.instancesHaveMember("missing")).isEqualTo(TriBool.FALSE);
  }

  @ParameterizedTest
  @ValueSource(strings = {"typing.final", "typing.disjoint_base"})
  void project_reexports_resolve_canonically(String marker) {
    int split = marker.lastIndexOf('.');
    var original = new FunctionDescriptor.FunctionDescriptorBuilder().withName(marker.substring(split + 1)).withFullyQualifiedName(marker).build();
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.typeShedDescriptorsProvider().descriptorsForModule(marker.substring(0, split)).put(original.name(), original);
    symbols.insertEntry("wrapper", Set.of(new AliasDescriptor("mark", "wrapper.mark", original)));
    var table = new ProjectLevelTypeTable(symbols);
    var type = type("from wrapper import mark\n@mark\nclass Marked: pass", table);
    assertThat(names(type)).containsExactly(marker);
    assertThat(check(type, table)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void repeated_class_declarations_keep_their_own_decorator_metadata() {
    var source = TypesTestUtils.parseAndInferTypes("""
      from typing import final
      @final
      class Marked: pass
      @custom
      class Marked: pass
      """);
    List<ClassDef> definitions = PythonTestUtils.getAllDescendant(source, tree -> tree.is(Tree.Kind.CLASSDEF));
    assertThat(definitions).hasSize(2);
    var first = (ClassType) definitions.get(0).name().typeV2();
    var second = (ClassType) definitions.get(1).name().typeV2();
    assertThat(names(first)).containsExactly("typing.final");
    assertThat(check(first)).isEqualTo(TriBool.TRUE);
    assertThat(names(second)).containsExactly("");
    assertThat(check(second)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void unions_of_known_marker_identities_are_class_preserving() {
    var type = type("from typing import final\nfrom typing_extensions import final as other\n@(final if condition else other)\nclass C: pass");
    assertThat(type.decorators()).hasSize(1);
    assertThat(type.decorators().get(0).expressionType().type()).isInstanceOf(UnionType.class);
    assertThat(check(type)).isEqualTo(TriBool.TRUE);
    var mixed = type("from typing import final\ndef custom(cls): return cls\n@(final if condition else custom)\nclass C: pass");
    assertThat(check(mixed)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void unresolved_import_paths_do_not_establish_marker_identity() {
    var unknownMarker = new UnknownType.UnresolvedImportType("typing.final");
    var type = new ClassTypeBuilder("C", "mod.C")
      .withDecorators(List.of(new DecoratorInfo(TypeWrapper.of(unknownMarker))))
      .build();
    assertThat(check(type)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void missing_marker_prevents_establishing_safety_after_export() {
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.insertEntry("typing", Set.of());
    var table = new ProjectLevelTypeTable(symbols);
    var type = type("from typing import disjoint_base\n@disjoint_base\nclass C: pass", table);
    assertThat(check(type, table)).isEqualTo(TriBool.FALSE);
    var descriptor = new PythonTypeToDescriptorConverter().convert("usage", new SymbolV2Impl("C"), Set.of(type));
    var restored = DescriptorsToProtobuf.fromProtobuf(DescriptorsToProtobuf.toProtobufModuleDescriptor(Set.of(descriptor)));
    symbols.insertEntry("usage", restored);
    var importedTable = new ProjectLevelTypeTable(symbols);
    assertThat(check(importedTable.getType("usage.C"), importedTable)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void conditional_decorator_identity_survives_export() {
    var type = type("from typing import final\n@(final if condition else final)\nclass C: pass");
    var descriptor = new PythonTypeToDescriptorConverter().convert("usage", new SymbolV2Impl("C"), Set.of(type));
    var symbols = ProjectLevelSymbolTable.empty();
    symbols.insertEntry("usage", Set.of(descriptor));
    var table = new ProjectLevelTypeTable(symbols);
    var imported = (ClassType) table.getType("usage.C");
    assertThat(imported.decorators().stream().map(DecoratorDescriptorConverter::toDescriptor))
      .containsExactlyElementsOf(((ClassDescriptor) descriptor).decorators());
    assertThat(check(imported, table)).isEqualTo(TriBool.TRUE);
  }

  private static ProjectLevelTypeTable markerTable(String marker) {
    int split = marker.lastIndexOf('.');
    var original = new FunctionDescriptor.FunctionDescriptorBuilder()
      .withName(marker.substring(split + 1)).withFullyQualifiedName(marker).build();
    var table = ProjectLevelSymbolTable.empty();
    // Verify the policy independently of whether the Typeshed pin already contains this marker.
    table.typeShedDescriptorsProvider().descriptorsForModule(marker.substring(0, split)).put(original.name(), original);
    return new ProjectLevelTypeTable(table);
  }

  private static ClassType type(String code, ProjectLevelTypeTable table) {
    var source = TypesTestUtils.parseAndInferTypes(table, PythonTestUtils.pythonFile("usage.py"), code);
    ClassDef definition = PythonTestUtils.getLastDescendant(source, tree -> tree.is(Tree.Kind.CLASSDEF));
    return (ClassType) definition.name().typeV2();
  }

  private static List<String> names(ClassType type) {
    return type.decorators().stream().map(decorator -> decorator.calleeType().isPresent() ? ""
      : FullyQualifiedNameHelper.getFullyQualifiedName(decorator.expressionType().type()).orElse("")).toList();
  }

  private TriBool check(PythonType type) {
    return check(type, TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE);
  }

  private TriBool check(PythonType type, ProjectLevelTypeTable table) {
    return TypePredicateUtils.evaluate(predicate, type, TypePredicateContext.of(table));
  }

  private static ClassType type(String code) {
    return (ClassType) TypesTestUtils.lastClassDef(code).name().typeV2();
  }
}
