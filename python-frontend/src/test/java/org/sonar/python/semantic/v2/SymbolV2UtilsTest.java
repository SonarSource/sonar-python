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
package org.sonar.python.semantic.v2;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;

class SymbolV2UtilsTest {

  @Test
  void is_function_or_class_declaration() {
    FileInput fileInput = parse("""
      def foo(): ...
      foo
      class MyClass: ...
      MyClass
      import bar
      bar
      """);
    new SymbolTableBuilderV2(fileInput).build();

    FunctionDef functionDef = (FunctionDef) fileInput.statements().statements().get(0);
    Name fooName = (Name) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0);
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(2);
    Name myClassName = (Name) ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0);
    ImportName importName = (ImportName) fileInput.statements().statements().get(4);
    Name barName = (Name) ((ExpressionStatement) fileInput.statements().statements().get(5)).expressions().get(0);

    UsageV2 functionDefUsage = functionDef.name().symbolV2().usages().stream().filter(u -> functionDef.name().equals(u.tree())).findFirst().get();
    UsageV2 nameUsage = fooName.symbolV2().usages().stream().filter(u -> fooName.equals(u.tree())).findFirst().get();
    UsageV2 classDefUsage = classDef.name().symbolV2().usages().stream().filter(u -> classDef.name().equals(u.tree())).findFirst().get();
    UsageV2 myClassaNameUsage = myClassName.symbolV2().usages().stream().filter(u -> myClassName.equals(u.tree())).findFirst().get();
    UsageV2 barImportUsage = importName.modules().get(0).dottedName().names().get(0).symbolV2().usages()
      .stream().filter(u -> importName.modules().get(0).dottedName().names().get(0).equals(u.tree())).findFirst().get();
    UsageV2 barNameUsage = barName.symbolV2().usages().stream().filter(u -> barName.equals(u.tree())).findFirst().get();


    assertThat(SymbolV2Utils.isDeclaration(functionDefUsage)).isTrue();
    assertThat(SymbolV2Utils.isDeclaration(nameUsage)).isFalse();
    assertThat(SymbolV2Utils.isDeclaration(classDefUsage)).isTrue();
    assertThat(SymbolV2Utils.isDeclaration(myClassaNameUsage)).isFalse();
    assertThat(SymbolV2Utils.isDeclaration(barImportUsage)).isTrue();
    assertThat(SymbolV2Utils.isDeclaration(barNameUsage)).isFalse();
  }
}
