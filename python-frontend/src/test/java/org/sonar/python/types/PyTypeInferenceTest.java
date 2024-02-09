/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.types;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolTableBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.pythonFile;

class PyTypeInferenceTest {

  @Test
  void test_type_inference_builtins() {
    TypeContext typeContext = TypeContext.fromJSON("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"x\",\n" +
      "      \"start_line\": 1,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"builtins.int\",\n" +
      "      \"short_type\": \"int\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"x\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"builtins.int\",\n" +
      "      \"short_type\": \"int\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    assertThat(lastExpression(typeContext, "x = 42",
      "x").type()).isEqualTo(InferredTypes.INT);
  }

  @Test
  void test_type_inference_custom_class() {
    TypeContext typeContext = TypeContext.fromJSON("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"a\",\n" +
      "      \"start_line\": 4,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"a\",\n" +
      "      \"start_line\": 5,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"A\",\n" +
      "      \"start_line\": 4,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(A),))\",\n" +
      "      \"short_type\": \"Type[A]\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    assertThat(lastExpression(typeContext, "class A:",
      "    def foo(self):",
      "        ...",
      "a = A()",
      "a").type().resolveMember("foo")).isPresent();
  }

  @Test
  void test_type_inference_with_scopes() {
    TypeContext typeContext = TypeContext.fromJSON("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"a\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"a\",\n" +
      "      \"start_line\": 4,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"a\",\n" +
      "      \"start_line\": 10,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"b\",\n" +
      "      \"start_line\": 11,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"b\",\n" +
      "      \"start_line\": 12,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"A\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(A),))\",\n" +
      "      \"short_type\": \"Type[A]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"a\",\n" +
      "      \"start_line\": 8,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"Outer.A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"a\",\n" +
      "      \"start_line\": 9,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"A\",\n" +
      "      \"short_type\": \"Outer.A\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"A\",\n" +
      "      \"start_line\": 11,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(A),))\",\n" +
      "      \"short_type\": \"Type[A]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"A\",\n" +
      "      \"start_line\": 8,\n" +
      "      \"start_col\": 8,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(Outer.A),))\",\n" +
      "      \"short_type\": \"Type[Outer.A]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"print\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 19,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), ClassType(builtins.NoneType)))\",\n" +
      "      \"short_type\": \"Callable[..., None]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"print\",\n" +
      "      \"start_line\": 7,\n" +
      "      \"start_col\": 23,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), ClassType(builtins.NoneType)))\",\n" +
      "      \"short_type\": \"Callable[..., None]\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    FileInput fileInput = getFileInputFromLines(typeContext, "class A:",
      "    def foo(self): print(\"foo\")",
      "a = A()",
      "a",
      "class Outer:",
      "    class A:",
      "        def bar(self): print(\"bar\")",
      "    a = A()",
      "    a",
      "a",
      "b = A()",
      "b");

    ExpressionStatement expressionStatement = ((ExpressionStatement) fileInput.statements().statements().get(2));
    assertThat(expressionStatement.expressions().get(0).type().resolveMember("foo")).isPresent();

    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(3);
    assertThat(((ExpressionStatement) classDef.body().statements().get(2)).expressions().get(0).type().resolveMember("bar")).isPresent();

    expressionStatement = ((ExpressionStatement) fileInput.statements().statements().get(4));
    assertThat(expressionStatement.expressions().get(0).type().resolveMember("foo")).isPresent();

    expressionStatement = ((ExpressionStatement) fileInput.statements().statements().get(6));
    assertThat(expressionStatement.expressions().get(0).type().resolveMember("foo")).isPresent();
  }

  private static FileInput getFileInputFromLines(TypeContext typeContext, String... lines) {
    String code = String.join("\n", lines);
    return PythonTestUtils.parse(new SymbolTableBuilder("",
      pythonFile("mod1.py"),
      ProjectLevelSymbolTable.empty(),
      typeContext),
      code);
  }
}
