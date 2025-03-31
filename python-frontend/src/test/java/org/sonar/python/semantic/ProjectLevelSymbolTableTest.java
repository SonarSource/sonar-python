/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.assertj.core.groups.Tuple;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.DeclaredType;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.semantic.ProjectLevelSymbolTable.empty;
import static org.sonar.python.semantic.ProjectLevelSymbolTable.from;

class ProjectLevelSymbolTableTest {

  private Map<String, Symbol> getSymbolByName(FileInput fileInput) {
    return fileInput.globalVariables().stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
  }

  @Test
  void wildcard_import() {
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod",
      Set.of(
        new VariableDescriptor("a", "mod.a", null),
        new VariableDescriptor("b", "mod.b", null),
        new ClassDescriptor.ClassDescriptorBuilder().withName("C").withFullyQualifiedName("mod.C").build()));

    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod import *",
      "print(a)"
    );
    assertThat(tree.globalVariables()).extracting(Symbol::name).containsExactlyInAnyOrder("a", "b", "C");
    Symbol a = getSymbolByName(tree).get("a");
    assertThat(a.fullyQualifiedName()).isEqualTo("mod.a");
    assertThat(a.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.OTHER, Usage.Kind.IMPORT);

    Symbol b = getSymbolByName(tree).get("b");
    assertThat(b.fullyQualifiedName()).isEqualTo("mod.b");
    assertThat(b.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT);

    Symbol c = getSymbolByName(tree).get("C");
    assertThat(c.fullyQualifiedName()).isEqualTo("mod.C");
    assertThat(c.usages()).extracting(Usage::kind).containsExactly(Usage.Kind.IMPORT);
  }

  @Test
  void unresolved_wildcard_import() {
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), empty()),
      "from external import *",
      "print(a)"
    );
    ImportFrom importFrom = ((ImportFrom) PythonTestUtils.getAllDescendant(tree, t -> t.is(Tree.Kind.IMPORT_FROM)).get(0));
    assertThat(importFrom.hasUnresolvedWildcardImport()).isTrue();
  }

  @Test
  void function_symbol() {
    FunctionDef functionDef = (FunctionDef) parse("def fn(p1, p2): pass").statements().statements().get(0);
    FunctionSymbolImpl fnSymbol = new FunctionSymbolImpl(functionDef, "mod.fn", pythonFile("mod.py"));

    var fnDescriptor = new FunctionDescriptor.FunctionDescriptorBuilder().withName("fn").withFullyQualifiedName("mod.fn").build();
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod", Set.of(fnDescriptor));

    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
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
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
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
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
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
  void submodule_import() {
    var fnDescriptor = new FunctionDescriptor.FunctionDescriptorBuilder().withName("fn").withFullyQualifiedName("mod.fn").build();
    var subModFnDescriptor = new FunctionDescriptor.FunctionDescriptorBuilder().withName("fn2").withFullyQualifiedName("mod.submod.fn2").build();
    Map<String, Set<Descriptor>> globalDescriptors = Map.of(
      "mod", Set.of(fnDescriptor),
      "mod.submod", Set.of(subModFnDescriptor));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "import mod.submod",
      "mod.submod.fn2(1, 2)");
    CallExpression callExpression = PythonTestUtils.getFirstChild(tree, t -> t.is(Tree.Kind.CALL_EXPR));
    Symbol importedFnSymbol = callExpression.calleeSymbol();
    assertThat(importedFnSymbol.is(Symbol.Kind.FUNCTION)).isTrue();
    assertThat(importedFnSymbol.fullyQualifiedName()).isEqualTo("mod.submod.fn2");
  }

  @Test
  void import_already_existing_symbol() {
    var fnDescriptor = new FunctionDescriptor.FunctionDescriptorBuilder().withName("fn").withFullyQualifiedName("mod.fn").build();
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod", Set.of(fnDescriptor));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "fn = 42",
      "from mod import fn"
    );
    assertThat(tree.globalVariables()).hasSize(1);
    Symbol importedFnSymbol = tree.globalVariables().iterator().next();
    assertThat(importedFnSymbol.kind()).isEqualTo(Symbol.Kind.OTHER);
    assertThat(importedFnSymbol.name()).isEqualTo("fn");
    assertThat(importedFnSymbol.fullyQualifiedName()).isNull();

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "mod = 42",
      "import mod"
    );
    assertThat(tree.globalVariables()).hasSize(1);
    Symbol modSymbol = tree.globalVariables().iterator().next();
    assertThat(modSymbol.kind()).isEqualTo(Symbol.Kind.OTHER);
    assertThat(modSymbol.name()).isEqualTo("mod");
    assertThat(modSymbol.fullyQualifiedName()).isNull();
  }

  @Test
  void other_imported_symbol() {
    VariableDescriptor xDescriptor = new VariableDescriptor("x", "mod.x", null);
    Set<Descriptor> modDescriptors = Set.of(xDescriptor);
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod", modDescriptors);
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod import x");
    Symbol importedXSymbol = tree.globalVariables().iterator().next();
    assertThat(importedXSymbol.name()).isEqualTo("x");
    assertThat(importedXSymbol.kind()).isEqualTo(Symbol.Kind.OTHER);
    assertThat(importedXSymbol.fullyQualifiedName()).isEqualTo("mod.x");
    assertThat(importedXSymbol.usages()).hasSize(1);
  }

  @Test
  void aliased_imported_symbols() {
    VariableDescriptor xDescriptor = new VariableDescriptor("x", "mod.x", null);
    Set<Descriptor> modDescriptors = Set.of(xDescriptor);
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod", modDescriptors);
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod import x as y");
    Symbol importedYSymbol = tree.globalVariables().iterator().next();
    assertThat(importedYSymbol.name()).isEqualTo("y");

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "import mod as mod1");
    Symbol importedModSymbol = tree.globalVariables().iterator().next();
    assertThat(importedModSymbol.name()).isEqualTo("mod1");
  }

  @Test
  void type_hierarchy() {
    var classADescriptor = new ClassDescriptor.ClassDescriptorBuilder().withName("A").withFullyQualifiedName("mod1.A").withSuperClasses(Set.of("mod2.B")).build();
    var classBDescriptor = new ClassDescriptor.ClassDescriptorBuilder().withName("B").withFullyQualifiedName("mod2.B").build();
    Map<String, Set<Descriptor>> globalDescriptors = new HashMap<>();
    globalDescriptors.put("mod1", Set.of(classADescriptor));
    globalDescriptors.put("mod2", Set.of(classBDescriptor));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod1 import A");
    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.name()).isEqualTo("A");
    assertThat(importedASymbol.fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbol classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "import mod1",
      "mod1.A"
    );

    QualifiedExpression qualifiedExpression = PythonTestUtils.getFirstChild(tree, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
    importedASymbol = qualifiedExpression.name().symbol();
    assertThat(importedASymbol.name()).isEqualTo("A");
    assertThat(importedASymbol.fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);
  }

  @Test
  void not_class_symbol_in_super_class() {
    var classADescriptor = new ClassDescriptor("A", "mod1.A", Collections.singleton("mod1.foo"), Set.of(), false, null, false, false, null, false);
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod1", Set.of(classADescriptor));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod1 import A"
    );

    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbol classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isTrue();
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.OTHER);
  }

  @Test
  void metaclass_in_imported_symbol() {
    var aClassDescriptor = new ClassDescriptor("A", "mod1.A", Collections.singleton("abc.ABCMeta"), Set.of(), false, null, false, true, "abc.ABCMeta", false);
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod1", Set.of(aClassDescriptor));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod1 import A"
    );

    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbolImpl classA = (ClassSymbolImpl) importedASymbol;
    assertThat(classA.hasMetaClass()).isTrue();
    assertThat(classA.metaclassFQN()).isEqualTo("abc.ABCMeta");
  }

  @Test
  void builtin_symbol_in_super_class() {
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod1",
      Collections.singleton(
        new ClassDescriptor.ClassDescriptorBuilder().withName("A").withFullyQualifiedName("mod1.A").withSuperClasses(Set.of("BaseException")).build()));
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod1 import A");

    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbol classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);
  }

  @Test
  void multi_level_type_hierarchy() {
    var classADescriptor = new ClassDescriptor.ClassDescriptorBuilder().withName("A").withFullyQualifiedName("mod1.A").withSuperClasses(Set.of("mod2.B")).build();
    var classBDescriptor = new ClassDescriptor.ClassDescriptorBuilder().withName("B").withFullyQualifiedName("mod2.B").withSuperClasses(Set.of("mod3.C")).build();
    var classCDescriptor = new ClassDescriptor.ClassDescriptorBuilder().withName("C").withFullyQualifiedName("mod3.C").build();

    Map<String, Set<Descriptor>> globalDescriptors = new HashMap<>();
    globalDescriptors.put("mod1", Set.of(classADescriptor));
    globalDescriptors.put("mod2", Set.of(classBDescriptor));
    globalDescriptors.put("mod3", Set.of(classCDescriptor));

    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod1 import A");
    Symbol importedASymbol = tree.globalVariables().iterator().next();
    assertThat(importedASymbol.name()).isEqualTo("A");
    assertThat(importedASymbol.fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    ClassSymbol classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);

    tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "import mod1",
      "mod1.A"
    );

    QualifiedExpression qualifiedExpression = PythonTestUtils.getFirstChild(tree, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
    importedASymbol = qualifiedExpression.name().symbol();
    assertThat(importedASymbol.name()).isEqualTo("A");
    assertThat(importedASymbol.fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(importedASymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    classA = (ClassSymbol) importedASymbol;
    assertThat(classA.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(classA.superClasses()).hasSize(1);
    assertThat(classA.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);

    ClassSymbol classB = (ClassSymbol) classA.superClasses().get(0);
    assertThat(classB.superClasses()).hasSize(1);
    assertThat(classB.superClasses().get(0).kind()).isEqualTo(Symbol.Kind.CLASS);
  }

  @Test
  void ambiguous_imported_symbol() {
    var ambiguousDescriptor = AmbiguousDescriptor.create(
      new FunctionDescriptor.FunctionDescriptorBuilder().withName("foo").withFullyQualifiedName("mod.foo").withParameters(
        List.of(
          new FunctionDescriptor.Parameter("a", null, false, false, false, false, false, null),
          new FunctionDescriptor.Parameter("b", null, false, false, false, false, false, null)))
        .build(),
      new FunctionDescriptor.FunctionDescriptorBuilder().withName("foo").withFullyQualifiedName("mod.foo").withParameters(
        List.of(
          new FunctionDescriptor.Parameter("a", null, false, false, false, false, false, null),
          new FunctionDescriptor.Parameter("b", null, false, false, false, false, false, null),
          new FunctionDescriptor.Parameter("c", null, false, false, false, false, false, null)))
        .build());
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod", Set.of(ambiguousDescriptor));

    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
      "from mod import foo"
    );
    Symbol importedFooSymbol = tree.globalVariables().iterator().next();
    assertThat(importedFooSymbol.name()).isEqualTo("foo");
    assertThat(importedFooSymbol.kind()).isEqualTo(Symbol.Kind.AMBIGUOUS);
    assertThat(importedFooSymbol.fullyQualifiedName()).isEqualTo("mod.foo");
    assertThat(importedFooSymbol.usages()).hasSize(1);
  }

  @Test
  void imported_class_hasSuperClassWithoutSymbol() {
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod",
      Set.of(new FunctionDescriptor.FunctionDescriptorBuilder().withName("foo").withFullyQualifiedName("mod.foo").build(),
        new ClassDescriptor.ClassDescriptorBuilder().withName("A").withFullyQualifiedName("mod.A").withSuperClasses(Set.of("foo")).build()));

    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), from(globalDescriptors)),
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
    return globalSymbols(empty(), fileInput, packageName);
  }

  private static Set<Symbol> globalSymbols(ProjectLevelSymbolTable projectLevelSymbolTable, FileInput fileInput, String packageName) {
    projectLevelSymbolTable.addModule(fileInput, packageName, pythonFile("mod.py"));
    return projectLevelSymbolTable.getSymbolsFromModule(packageName.isEmpty() ? "mod" : packageName + ".mod");
  }

  @Test
  void test_remove_module() {
    FileInput tree = parseWithoutSymbols(
      "class A: pass"
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = empty();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).extracting(Symbol::name).containsExactlyInAnyOrder("A");
    projectLevelSymbolTable.removeModule("", "mod.py");
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).isNull();
  }

  @Test
  void test_insert_entry() {
    ProjectLevelSymbolTable projectLevelSymbolTable = empty();
    VariableDescriptor variableDescriptor = new VariableDescriptor("x", "mod.x", null);
    projectLevelSymbolTable.insertEntry("mod", Set.of(variableDescriptor));
    assertThat(projectLevelSymbolTable.descriptorsForModule("mod")).containsExactly(variableDescriptor);
    assertThat(projectLevelSymbolTable.getSymbol("mod.x").name()).isEqualTo("x");
  }

  @Test
  void test_add_module_after_creation() {
    FileInput tree = parseWithoutSymbols(
      "class A: pass"
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = empty();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).extracting(Symbol::name).containsExactlyInAnyOrder("A");
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod2")).isNull();
    assertThat(projectLevelSymbolTable.getSymbol("mod.A")).isNotNull();
    assertThat(projectLevelSymbolTable.getSymbol("mod2.B")).isNull();

    tree = parseWithoutSymbols(
      "class B: pass"
    );
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod2.py"));
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).extracting(Symbol::name).containsExactlyInAnyOrder("A");
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod2")).extracting(Symbol::name).containsExactlyInAnyOrder("B");
    assertThat(projectLevelSymbolTable.getSymbol("mod2.B")).isNotNull();
  }

  @Test
  void test_imported_modules() {
    FileInput tree = parseWithoutSymbols(
      "import A"
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = empty();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));
    assertThat(projectLevelSymbolTable.importsByModule()).containsExactly(Map.entry("mod", Set.of("A")));
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod2")).isNull();
    assertThat(projectLevelSymbolTable.getSymbol("mod.A")).isNull();
    assertThat(projectLevelSymbolTable.getSymbol("mod2.B")).isNull();

    tree = parseWithoutSymbols(
      "import A, B"
    );
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod2.py"));
    assertThat(projectLevelSymbolTable.importsByModule()).containsOnly(Map.entry("mod", Set.of("A")), Map.entry("mod2", Set.of("A", "B")));
    assertThat(projectLevelSymbolTable.getSymbol("mod2.B")).isNull();

    tree = parseWithoutSymbols(
      "from C.D import foo, bar"
    );
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod3.py"));
    assertThat(projectLevelSymbolTable.importsByModule()).containsOnly(
      Map.entry("mod", Set.of("A")),
      Map.entry("mod2", Set.of("A", "B")),
      Map.entry("mod3", Set.of("C.D"))
    );
    assertThat(projectLevelSymbolTable.getSymbol("mod2.B")).isNull();


    tree = parseWithoutSymbols(
      "from E import *"
    );
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod4.py"));
    assertThat(projectLevelSymbolTable.importsByModule()).containsOnly(
      Map.entry("mod", Set.of("A")),
      Map.entry("mod2", Set.of("A", "B")),
      Map.entry("mod3", Set.of("C.D")),
      Map.entry("mod4", Set.of("E"))
    );

    tree = parseWithoutSymbols(
      "from ..F import G"
    );
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod5.py"));
    assertThat(projectLevelSymbolTable.importsByModule()).containsOnly(
      Map.entry("mod", Set.of("A")),
      Map.entry("mod2", Set.of("A", "B")),
      Map.entry("mod3", Set.of("C.D")),
      Map.entry("mod4", Set.of("E")),
      Map.entry("mod5", Set.of("F"))
    );

    tree = parseWithoutSymbols(
      "from ..F import G"
    );
    projectLevelSymbolTable.addModule(tree, "my_package.my_subpackage", pythonFile("mod6.py"));
    assertThat(projectLevelSymbolTable.importsByModule()).containsOnly(
      Map.entry("mod", Set.of("A")),
      Map.entry("mod2", Set.of("A", "B")),
      Map.entry("mod3", Set.of("C.D")),
      Map.entry("mod4", Set.of("E")),
      Map.entry("mod5", Set.of("F")),
      Map.entry("my_package.my_subpackage.mod6", Set.of("my_package.F"))
    );
  }

  @Test
  void importedStubModules() {
    FileInput tree = parseWithoutSymbols("""
      import math
      import os
      """);
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    projectLevelSymbolTable.addModule(tree, "my_package", pythonFile("mod.py"));
    assertThat(projectLevelSymbolTable.typeShedDescriptorsProvider().stubModules()).containsExactlyInAnyOrder("math", "os", "django", "django.urls.conf", "django.urls");
  }

  @Test
  void global_symbols() {
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
  void global_symbols_override() {
    var tree = parseWithoutSymbols(
      """
      obj1 = 42
      obj2: int = 42
      def fn(): pass
      class A: pass
      """
    );
    var symbolTable = empty();
    var globalSymbols = globalSymbols(symbolTable, tree, "");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("obj1", "obj2", "fn", "A");
    assertThat(globalSymbols).extracting(Symbol::fullyQualifiedName).containsExactlyInAnyOrder("mod.obj1", "mod.obj2", "mod.fn", "mod.A");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());

    tree = parseWithoutSymbols(
      """
      obj1 = 42
      obj2: int = 42
      def fn(): pass
      class B: pass
      """
    );
    globalSymbols = globalSymbols(symbolTable, tree, "");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("obj1", "obj2", "fn", "A", "B");
    assertThat(globalSymbols).extracting(Symbol::fullyQualifiedName).containsExactlyInAnyOrder("mod.obj1", "mod.obj2", "mod.fn", "mod.A", "mod.B");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());

    var nonAmbiguousSymbolFqn = Set.of("mod.A", "mod.B");

    assertThat(globalSymbols)
      .filteredOn(s -> !nonAmbiguousSymbolFqn.contains(s.fullyQualifiedName()))
      .allMatch(AmbiguousSymbol.class::isInstance);

    assertThat(globalSymbols)
      .filteredOn(s -> nonAmbiguousSymbolFqn.contains(s.fullyQualifiedName()))
      .allMatch(ClassSymbol.class::isInstance);
  }

  @Test
  void global_symbols_private_by_convention() {
    // although being private by convention, it's considered as exported
    FileInput tree = parseWithoutSymbols(
      "def _private_fn(): pass"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("_private_fn");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  void local_symbols_not_exported() {
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
  void redefined_symbols() {
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
  void function_symbols() {
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
  void redefined_class_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "C = \"hello\"",
      "class C: ",
      "  pass");
    Set<Symbol> globalSymbols = globalSymbols(fileInput, "mod");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("C");
    assertThat(globalSymbols).extracting(Symbol::kind).allSatisfy(k -> assertThat(k).isEqualTo(Symbol.Kind.CLASS));
  }

  @Test
  @Disabled("SONARPY-2248")
  void classdef_with_missing_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "class C: ",
      "  pass",
      "global C");

    Set<Symbol> globalSymbols = globalSymbols(fileInput, "mod");
    assertThat(globalSymbols).isNotEmpty();
  }

  @Test
  void class_symbol() {
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
  }

  @Test
  void class_symbol_inheritance_from_nested_class() {
    // for the time being, we only consider symbols defined in the global scope
    var fileInput = parseWithoutSymbols(
      "class A:",
      "  class A1: pass",
      "class C(A.A1): ",
      "  pass");
    var globalSymbols = globalSymbols(fileInput, "mod");
    var symbols = globalSymbols.stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    var cSymbol = symbols.get("C");
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).hasSize(1);
  }

  @Test
  void child_class_method_call_is_not_a_member_of_parent_class() {
    ProjectLevelSymbolTable projectLevelSymbolTable = empty();
    FileInput importedFileInput = parseWithoutSymbols(
      "class A:",
      "  def meth(self): ",
      "    return self.foo()"
    );
    FileInput importFileInput = parseWithoutSymbols(
      "from mod import A",
      "class B(A): ",
      "  def foo(self):",
      "    pass"
    );
    projectLevelSymbolTable.addModule(importFileInput, "packageName", pythonFile("mod2.py"));
    projectLevelSymbolTable.addModule(importedFileInput, "packageName", pythonFile("mod.py"));
    Set<Symbol> globalSymbols = projectLevelSymbolTable.getSymbolsFromModule("packageName.mod");
    // SONARPY-2327 The method call to foo() in class A is not a member of ClassSymbol A because the symbol is created from the ClassType through the Descriptor
    Optional<ClassSymbol> classA = globalSymbols.stream().filter(s -> s.name().equals("A")).map(ClassSymbol.class::cast).findFirst();
    assertThat(classA).isPresent();
    assertThat(classA.get().canHaveMember("foo")).isFalse();
    assertThat(classA.get().declaredMembers()).extracting("kind", "name").containsExactlyInAnyOrder(Tuple.tuple(Symbol.Kind.FUNCTION, "meth"));
  }

  @Test
  void class_inheriting_from_imported_symbol() {
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
  void symbol_duplicated_by_wildcard_import() {
    FileInput tree = parseWithoutSymbols(
      "def nlargest(n, iterable): ...",
      "from _heapq import *",
      "def nlargest(n, iterable, key=None): ..."
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).hasOnlyElementsOfType(FunctionSymbol.class);

    tree = parseWithoutSymbols(
      "nonlocal nlargest",
      "from _heapq import *",
      "def nlargest(n, iterable, key=None): ..."
    );
    globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  void class_having_itself_as_superclass_should_not_trigger_error() {
    FileInput fileInput = parseWithoutSymbols("class A(A): pass");
    Set<Symbol> globalSymbols = globalSymbols(fileInput, "mod");
    ClassSymbol a = (ClassSymbol) globalSymbols.iterator().next();
    // SONARPY-1350: The parent "A" is not yet defined  at the time it is read, so this is actually not correct
    assertThat(a.superClasses()).isEmpty();
    assertThat(a.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  void class_having_another_class_with_same_name_should_not_trigger_error() {
    FileInput fileInput = parseWithoutSymbols(
      "from external import B",
      "class A:",
      "  class B(B): pass"
    );
    Set<Symbol> symbols = globalSymbols(fileInput, "mod");
    // SONARPY-1829: Parent should be external.B
    ClassSymbol outerClassSymbol = ((ClassSymbol) symbols.stream().findFirst().get());
    ClassSymbol innerClassSymbol = (ClassSymbol) outerClassSymbol.resolveMember("B").get();
    assertThat(innerClassSymbol.superClasses()).isEmpty();
    assertThat(innerClassSymbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  void symbols_with_missing_type_are_not_exported() {
    FileInput fileInput = parseWithoutSymbols("""
      builtin_str = str
      str = str
      """
    );
    Set<Symbol> symbols = globalSymbols(fileInput, "my_package");
    assertThat(symbols).isEmpty();
  }

  @Test
  void global_symbols_stdlib_imports() {
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
  void module_importing_itself() {
    FileInput tree = parseWithoutSymbols(
      "from mod import *",
      "from mod import smth"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  void imports_wont_trigger_typeshed_lookup() {
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
  void loop_in_class_inheritance() {
    String[] foo = {
      "from bar import B",
      "class A(B): ..."
    };
    String[] bar = {
      "from foo import A",
      "class B(A): ..."
    };

    ProjectLevelSymbolTable projectSymbolTable = empty();
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
      .containsExactly(tuple(Symbol.Kind.CLASS, "foo.A"));
  }

  @Test
  void annotated_parameter_is_translated_correctly() {
    FileInput tree = parseWithoutSymbols(
      "def fn(param: str, *my_tuple, **my_dict): ..."
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactly("fn");
    FunctionSymbol functionSymbol = ((FunctionSymbol) globalSymbols.stream().findFirst().get());
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::declaredType).containsExactly(InferredTypes.DECL_STR, InferredTypes.TUPLE, InferredTypes.DICT);
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  void symbols_from_module_should_be_the_same() {
    FileInput tree = parseWithoutSymbols(
      "class A: ...",
             "class B(A): ..."
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = empty();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));
    Set<Symbol> mod = projectLevelSymbolTable.getSymbolsFromModule("mod");
    assertThat(mod).extracting(Symbol::name).containsExactlyInAnyOrder("A", "B");
    ClassSymbol classSymbolA = (ClassSymbol) mod.stream().filter(s -> s.fullyQualifiedName().equals("mod.A")).findFirst().get();
    ClassSymbol classSymbolB = (ClassSymbol) mod.stream().filter(s -> s.fullyQualifiedName().equals("mod.B")).findFirst().get();
    ClassSymbol superClass = (ClassSymbol) classSymbolB.superClasses().get(0);
    assertThat(superClass).isSameAs(classSymbolA);
  }

  @Test
  void imported_typeshed_symbols_are_not_exported() {
    FileInput tree = parseWithoutSymbols(
      "from html import escape"
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  void class_with_method_parameter_of_same_type() {
    FileInput tree = parseWithoutSymbols(
      "class Document:",
             "  def my_method(param: Document): ..."
    );
    Set<Symbol> globalSymbols = globalSymbols(tree, "");
    assertThat(globalSymbols).hasSize(1);
    ClassSymbol classSymbol = (ClassSymbol) globalSymbols.stream().findFirst().get();
    assertThat(classSymbol.declaredMembers()).hasSize(1);
    FunctionSymbol functionSymbol = (FunctionSymbol) classSymbol.declaredMembers().stream().findFirst().get();
    assertThat(functionSymbol.parameters()).hasSize(1);
    FunctionSymbol.Parameter parameter = functionSymbol.parameters().get(0);
    DeclaredType declaredType = (DeclaredType) parameter.declaredType();
    assertThat(declaredType.getTypeClass()).isSameAs(classSymbol);
  }

  @Test
  void no_stackoverflow_for_ambiguous_descriptor() {
    TypeShed.resetBuiltinSymbols();
    String[] foo = {
    "if cond:",
    "  Ambiguous = 41",
    "else:",
    "  class Ambiguous(SomeParent):",
    "    local_var = 'i'",
    "    def func(param: Ambiguous):",
    "      ..."
    };
    String[] bar = {
      "from foo import *\n",
    };
    ProjectLevelSymbolTable projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(foo), "", pythonFile("foo.py"));
    projectSymbolTable.addModule(parseWithoutSymbols(bar), "", pythonFile("bar.py"));

    Set<Symbol> fooSymbols = projectSymbolTable.getSymbolsFromModule("foo");
    assertThat(fooSymbols).hasSize(1);
    AmbiguousSymbol symbolFromProjectTable = ((AmbiguousSymbol) fooSymbols.stream().findFirst().get());
    assertThat(symbolFromProjectTable.fullyQualifiedName()).isEqualTo("foo.Ambiguous");
    assertThat(symbolFromProjectTable.alternatives()).hasSize(2);
    assertThat(symbolFromProjectTable.alternatives()).extracting(Symbol::kind).containsExactlyInAnyOrder(Symbol.Kind.CLASS, Symbol.Kind.OTHER);
    ClassSymbol classSymbol = (ClassSymbol) symbolFromProjectTable.alternatives().stream().filter(s -> s.kind().equals(Symbol.Kind.CLASS)).findFirst().get();
    assertThat(classSymbol.declaredMembers()).hasSize(2);
    assertThat(classSymbol.declaredMembers()).extracting(Symbol::kind).containsExactlyInAnyOrder(Symbol.Kind.FUNCTION, Symbol.Kind.OTHER);

    FileInput tree = parse(new SymbolTableBuilder("", pythonFile("bar.py"), projectSymbolTable), bar);
    assertThat(tree.globalVariables()).hasSize(1);
    AmbiguousSymbol localSymbol = (AmbiguousSymbol) tree.globalVariables().stream().findFirst().get();
    assertThat(localSymbol.fullyQualifiedName()).isEqualTo("foo.Ambiguous");
    assertThat(localSymbol.alternatives()).hasSize(2);
    assertThat(localSymbol.alternatives()).extracting(Symbol::kind).containsExactlyInAnyOrder(Symbol.Kind.CLASS, Symbol.Kind.OTHER);
  }

  @Test
  void ambiguous_descriptor_alternatives_dont_rely_on_FQN_for_conversion() {
    ProjectLevelSymbolTable projectLevelSymbolTable = empty();
    Set<Descriptor> descriptors = new HashSet<>();
    VariableDescriptor variableDescriptor = new VariableDescriptor("Ambiguous", "foo.Ambiguous", null);
    ClassDescriptor classDescriptor = new ClassDescriptor("Ambiguous", "foo.Ambiguous", List.of(), Set.of(), false, null, false, false, null, false);
    AmbiguousDescriptor ambiguousDescriptor = new AmbiguousDescriptor("Ambiguous", "foo.Ambiguous", Set.of(variableDescriptor, classDescriptor));
    descriptors.add(ambiguousDescriptor);

    projectLevelSymbolTable.insertEntry("foo", descriptors);
    Set<Symbol> foo = projectLevelSymbolTable.getSymbolsFromModule("foo");

    assertThat(foo).hasSize(1);
    Symbol retrievedSymbol = foo.stream().findFirst().get();
    assertThat(retrievedSymbol.kind()).isEqualTo(Symbol.Kind.AMBIGUOUS);
    AmbiguousSymbol ambiguousSymbol = (AmbiguousSymbol) retrievedSymbol;
    assertThat(ambiguousSymbol.alternatives()).hasSize(2);
  }

  @Test
  void loop_in_inheritance_with_method_paraneters_of_same_type() {
    String[] foo = {
      "from bar import B",
      "class A(B):",
      "  def my_A_method(param: A): ...",
      "  def my_A_other_method(param: B): ..."
    };
    String[] bar = {
      "from foo import A",
      "class B(A):",
      "  def my_B_method(param: A): ...",
      "  def my_B_other_method(param: B): ..."
    };

    ProjectLevelSymbolTable projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(foo), "", pythonFile("foo.py"));
    projectSymbolTable.addModule(parseWithoutSymbols(bar), "", pythonFile("bar.py"));

    Set<Symbol> fooSymbols = projectSymbolTable.getSymbolsFromModule("foo");
    ClassSymbol classSymbolA = (ClassSymbol) fooSymbols.stream().filter(s -> s.fullyQualifiedName().equals("foo.A")).findFirst().get();
    ClassSymbol classSymbolB = (ClassSymbol) classSymbolA.superClasses().get(0);

    assertThat(classSymbolB.superClasses()).containsExactly(classSymbolA);
    assertThat(classSymbolA.declaredMembers().stream()
      .map(FunctionSymbol.class::cast)
      .flatMap(f -> f.parameters().stream()
        .map(FunctionSymbol.Parameter::declaredType)
        .map(DeclaredType.class::cast))
      .toList())
      .extracting(DeclaredType::getTypeClass)
      .containsExactlyInAnyOrder(classSymbolA, classSymbolB);
  }

  @Test
  void django_views() {
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

    ProjectLevelSymbolTable projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(urls), "", pythonFile("urls.py"));

    assertThat(projectSymbolTable.isDjangoView("views.foo")).isTrue();

    FileInput fileInput = parse(new SymbolTableBuilder("", pythonFile("views.py"), projectSymbolTable), views);
    Map<String, Symbol> symbolByName = getSymbolByName(fileInput);
    FunctionSymbolImpl foo = ((FunctionSymbolImpl) symbolByName.get("foo"));
    FunctionSymbolImpl bar = ((FunctionSymbolImpl) symbolByName.get("bar"));
    assertThat(foo.isDjangoView()).isTrue();
    assertThat(bar.isDjangoView()).isFalse();
  }

  @Test
  void django_views_local_functions() {
    String content = """
      from django.urls import path
      
      def foo(): ...
      urlpatterns = [path('foo', foo, name='foo')]
      
      class MyClass:
        def bar(): ...
      
      urlpatterns.append(path('bar', MyClass.bar, name='bar'))
      
      class MyOtherClass:
        class MyNestedClass:
          def qix(): ...
      urlpatterns.append(path('bar', MyOtherClass.MyNestedClass.qix, name='bar'))
      """;

    ProjectLevelSymbolTable projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(content), "my_package", pythonFile("urls.py"));
    assertThat(projectSymbolTable.isDjangoView("my_package.urls.foo")).isTrue();
    assertThat(projectSymbolTable.isDjangoView("my_package.urls.MyClass.bar")).isTrue();
    assertThat(projectSymbolTable.isDjangoView("my_package.urls.MyOtherClass.MyNestedClass.qix")).isTrue();
  }

  @Test
  void django_views_ambiguous() {
    String content = """
      from django.urls import path
      if x:
        def ambiguous(): ...
      else:
        def ambiguous(): ...
      urlpatterns = [path('bar', ambiguous, name='bar')]
      """;
    ProjectLevelSymbolTable projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(content), "my_package", pythonFile("urls.py"));
    assertThat(projectSymbolTable.isDjangoView("my_package.urls.ambiguous")).isFalse();
  }

  @Test
  void django_views_conf_import() {
    String content = """
      from django.urls import conf
      import views
      urlpatterns = [conf.path('foo', views.foo, name='foo'), conf.path('baz')]
      """;
    ProjectLevelSymbolTable projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(content), "my_package", pythonFile("urls.py"));
    assertThat(projectSymbolTable.isDjangoView("views.foo")).isTrue();
  }

  @Test
  void django_views_same_class() {
    String content = """
      from django.urls import path

      class ClassWithViews:
        def view_method(self):
          ...

        def get_urlpatterns(self):
          return [path("something", self.view_method, name="something")]
      """;
    ProjectLevelSymbolTable projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(content), "my_package", pythonFile("mod.py"));
    // SONARPY-2322: should be true
    assertThat(projectSymbolTable.isDjangoView("my_package.mod.ClassWithViews.view_method")).isFalse();
  }

  /**
   * The variable `foo` which is assigned in the decorator of the function should belong to the global scope not the function scope
   */
  @Test
  void function_decorator_symbol() {
    FileInput fileInput = PythonTestUtils.parse(
      "@foo := bar",
      "def function(): pass"
    );
    assertThat(fileInput.globalVariables()).extracting(Symbol::name)
      .containsExactlyInAnyOrder("foo", "function");

    FunctionDef functionDef = (FunctionDef) fileInput.statements().statements().get(0);
    assertThat(functionDef.localVariables()).isEmpty();
  }

  @Test
  void superclasses_without_descriptor() {
    var code = """
      class MetaField: ...
      class Field(MetaField()): ...
      """;

    var projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(code), "", pythonFile("mod.py"));

    var descriptors = projectSymbolTable.getDescriptorsFromModule("mod");
    assertThat(descriptors).hasSize(2);

    var fieldClassDescriptor = descriptors.stream().filter(d -> "Field".equals(d.name()))
      .map(ClassDescriptor.class::cast)
      .findFirst()
      .orElse(null);
    assertThat(fieldClassDescriptor).isNotNull();
    assertThat(fieldClassDescriptor.superClasses()).isEmpty();
    assertThat(fieldClassDescriptor.hasSuperClassWithoutDescriptor()).isTrue();
    var symbol = (ClassSymbol) projectSymbolTable.getSymbol("mod.Field");
    assertThat(symbol.superClasses()).isEmpty();
    assertThat(symbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  void superclasses_without_descriptor_unresolved_import() {
    var code = """
      from unknown import MetaField
      class Field(MetaField): ...
      """;

    var projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(code), "", pythonFile("mod.py"));
    var symbol = (ClassSymbol) projectSymbolTable.getSymbol("mod.Field");
    assertThat(symbol.hasUnresolvedTypeHierarchy()).isTrue();
  }

  @Test
  void class_wth_imported_metaclass() {
    var code = """
      from abc import ABCMeta
      class WithMetaclass(metaclass=ABCMeta): ...
      """;

    var projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(code), "", pythonFile("mod.py"));
    var symbol = (ClassSymbolImpl) projectSymbolTable.getSymbol("mod.WithMetaclass");
    assertThat(symbol.metaclassFQN()).isEqualTo("abc.ABCMeta");
  }

  @Test
  void class_wth_locally_defined_metaclass() {
    var code = """
      class LocalMetaClass: ...
      class WithMetaclass(metaclass=LocalMetaClass): ...
      """;

    var projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(code), "", pythonFile("mod.py"));
    var symbol = (ClassSymbolImpl) projectSymbolTable.getSymbol("mod.WithMetaclass");
    assertThat(symbol.metaclassFQN()).isEqualTo("mod.LocalMetaClass");
  }

  @Test
  void class_wth_unresolved_import_metaclass() {
    var code = """
      from unknown import UnresolvedMetaClass
      class WithMetaclass(metaclass=UnresolvedMetaClass): ...
      """;

    var projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(code), "", pythonFile("mod.py"));
    var symbol = (ClassSymbolImpl) projectSymbolTable.getSymbol("mod.WithMetaclass");
    assertThat(symbol.metaclassFQN()).isEqualTo("unknown.UnresolvedMetaClass");
  }

  @Test
  void class_wth_call_result_metaclass() {
    var code = """
      def foo(): ...
      class WithMetaclass(metaclass=foo()): ...
      """;

    var projectSymbolTable = empty();
    projectSymbolTable.addModule(parseWithoutSymbols(code), "", pythonFile("mod.py"));
    var symbol = (ClassSymbolImpl) projectSymbolTable.getSymbol("mod.WithMetaclass");
    assertThat(symbol.hasMetaClass()).isTrue();
    assertThat(symbol.metaclassFQN()).isNull();
  }

  @Test
  void projectPackages() {
    ProjectLevelSymbolTable projectLevelSymbolTable = empty();
    projectLevelSymbolTable.addProjectPackage("first.package");
    projectLevelSymbolTable.addProjectPackage("second.package");
    projectLevelSymbolTable.addProjectPackage("third");
    assertThat(projectLevelSymbolTable.projectBasePackages()).containsExactlyInAnyOrder("first", "second", "third");
  }

}
