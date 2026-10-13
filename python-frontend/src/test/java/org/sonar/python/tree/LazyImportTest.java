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
package org.sonar.python.tree;

import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.ImportStatement;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.parser.PythonParser;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

class LazyImportTest {

  @ParameterizedTest
  @MethodSource("restrictedContexts")
  void restricted_contexts_preserve_imports_and_the_rest_of_the_file(String code) {
    // Invalid Python syntax: lazy imports cannot appear in functions, classes or try suites.
    // Preserve these import nodes so that a future rule can report placement without losing the file's AST.
    for (boolean notebook : new boolean[] {false, true}) {
      var statement = recoveredImport(code, notebook);
      if (statement instanceof ImportName name) {
        assertThat(name.modules().get(0).dottedName().names().get(0).name()).isEqualTo("json");
        assertThat(name.modules().get(0).alias().name()).isEqualTo("data");
      } else {
        var from = (ImportFrom) statement;
        assertThat(from.module().names().get(0).name()).isEqualTo("pathlib");
        assertThat(from.importedNames().get(0).alias().name()).isEqualTo("P");
      }
      assertThatCode(() -> parse(code.replace("lazy ", ""), notebook)).doesNotThrowAnyException();
    }
  }

  static Stream<String> restrictedContexts() {
    return withImportStatements(Stream.of(
      "def function():\n  %s\n",
      "async def function():\n  %s\n",
      "class Example:\n  %s\n",
      "def function():\n  if flag:\n    %s\n",
      "async def function():\n  with context:\n    %s\n",
      "class Example:\n  if flag:\n    %s\n",
      "try:\n  %s\nexcept ImportError:\n  pass\n",
      "try:\n  if flag:\n    %s\nexcept ImportError:\n  pass\n",
      "try:\n  pass\nexcept ImportError:\n  %s\n",
      "try:\n  pass\nexcept ImportError:\n  pass\nelse:\n  %s\n",
      "try:\n  pass\nfinally:\n  %s\n",
      "try:\n  pass\nexcept* ImportError:\n  %s\n"));
  }

  @ParameterizedTest
  @MethodSource("moduleContexts")
  void accepts_module_contexts(String code) {
    assertValid(code);
  }

  static Stream<String> moduleContexts() {
    return withImportStatements(Stream.of(
      "%s\n", "import os; %s\n",
      "if TYPE_CHECKING:\n  %s\n",
      "with context:\n  %s\n",
      "for item in items:\n  %s\n",
      "while flag:\n  %s\n",
      "try:\n  pass\nexcept ImportError:\n  pass\n%s\n"));
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "lazy from __future__.something import name", "lazy from __future__.something import (name as alias)",
    "lazy from .__future__ import annotations", "lazy from ..__future__ import (annotations as alias)",
    "lazy from __future__x import name", "lazy import __future__", "lazy = 42\ndef function():\n  lazy()"
  })
  void accepts_dotted_and_relative_future_modules_and_soft_keyword_uses(String code) {
    assertValid(code);
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "lazy from __future__ import annotations", "lazy from __future__ import (annotations as alias)",
    "lazy from __future__ import annotations as alias", "lazy from package import *", "lazy from .package import *"
  })
  void future_and_star_imports_preserve_imports_and_the_rest_of_the_file(String code) {
    // Invalid Python syntax: lazy future imports and lazy star imports are forbidden.
    // Their existing AST representation is still useful for analysis and later validity diagnostics.
    for (boolean notebook : new boolean[] {false, true}) {
      var statement = (ImportFrom) recoveredImport(code, notebook);
      assertThat(statement.isWildcardImport()).isEqualTo(code.endsWith("*"));
      assertThat(statement.dottedPrefixForModule()).hasSize(code.contains(".package") ? 1 : 0);
      if (statement.isWildcardImport()) {
        assertThat(statement.wildcard().value()).isEqualTo("*");
      } else {
        assertThat(statement.module().names().get(0).name()).isEqualTo("__future__");
        assertThat(statement.importedNames().get(0).dottedName().names().get(0).name()).isEqualTo("annotations");
        if (code.contains(" as alias")) {
          assertThat(statement.importedNames().get(0).alias().name()).isEqualTo("alias");
        }
      }
    }
  }

  private static Stream<String> withImportStatements(Stream<String> contexts) {
    return contexts.flatMap(context -> Stream.of("lazy import json as data", "lazy from pathlib import (Path as P)").map(context::formatted));
  }

  private static void assertValid(String code) {
    for (boolean notebook : new boolean[] {false, true}) {
      assertThatCode(() -> parse(code, notebook)).doesNotThrowAnyException();
    }
  }

  private static ImportStatement recoveredImport(String code, boolean notebook) {
    var file = parse(code + "\nsentinel\n", notebook);
    var statements = file.statements().statements();
    assertThat(statements.get(statements.size() - 1).firstToken().value()).isEqualTo("sentinel");
    ImportStatement statement = PythonTestUtils.getLastDescendant(file, ImportStatement.class::isInstance);
    var lazyKeyword = statement instanceof ImportName name ? name.lazyKeyword() : ((ImportFrom) statement).lazyKeyword();
    assertThat(lazyKeyword).isNotNull();
    assertThat(statement.firstToken()).isSameAs(lazyKeyword);
    assertThat(statement.parent()).isNotNull();
    assertThat(lazyKeyword.line()).isEqualTo((int) code.substring(0, code.indexOf("lazy ")).chars().filter(c -> c == '\n').count() + 1);
    return statement;
  }

  private static FileInput parse(String code, boolean notebook) {
    var parser = notebook ? PythonParser.createIPythonParser() : PythonParser.create();
    PythonTreeMaker treeMaker = notebook ? new IPythonTreeMaker() : new PythonTreeMaker();
    return treeMaker.fileInput(parser.parse(code));
  }
}
