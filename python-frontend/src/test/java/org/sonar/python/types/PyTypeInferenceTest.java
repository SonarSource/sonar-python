/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import javax.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.types.pytype.json.TypeContextReader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.pythonFile;

class PyTypeInferenceTest {

  @Test
  void test_type_inference_builtins() {
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
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
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
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
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
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

  @Test
  void test_import_of_known_typeshed_symbol() {
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"connection\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"http.client.HTTPConnection\",\n" +
      "      \"short_type\": \"http.client.HTTPConnection\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"connection\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"http.client.HTTPConnection\",\n" +
      "      \"short_type\": \"http.client.HTTPConnection\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"HTTPConnection\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 13,\n" +
      "      \"syntax_role\": \"Attribute\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(http.client.HTTPConnection),))\",\n" +
      "      \"short_type\": \"Type[http.client.HTTPConnection]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"client\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 13,\n" +
      "      \"syntax_role\": \"Attribute\",\n" +
      "      \"type\": \"Alias(name='http.client', type=Module(name='http.client', module_name='http.client'))\",\n" +
      "      \"short_type\": \"module\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"http\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 13,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"Alias(name='http', type=Module(name='http', module_name='http'))\",\n" +
      "      \"short_type\": \"module\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    /*
     * SNIPPET
     *
     * import http
     * connection = http.client.HTTPConnection('www.python.org')
     * connection <- connection should have the appropriate type
     *
     */

    FileInput fileInput = getFileInputFromLines(typeContext, "import http",
      "connection = http.client.HTTPConnection('www.python.org')",
      "connection");

    ExpressionStatement expressionStatement = ((ExpressionStatement) fileInput.statements().statements().get(2));
    InferredType inferredType = expressionStatement.expressions().get(0).type();
    // assertThat(inferredType.resolveMember("set_tunnel")).isPresent(); // This fails at this point. It should be fixed.
    assertThat(inferredType.runtimeTypeSymbol().name()).isEqualTo("HTTPConnection");
    assertThat(inferredType.runtimeTypeSymbol().fullyQualifiedName()).isEqualTo("http.client.HTTPConnection");
  }

  @Test
  void test_import_of_known_typeshed_symbol_2() {
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"connection\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"http.client.HTTPConnection\",\n" +
      "      \"short_type\": \"http.client.HTTPConnection\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"connection\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"http.client.HTTPConnection\",\n" +
      "      \"short_type\": \"http.client.HTTPConnection\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"HTTPConnection\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 13,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(http.client.HTTPConnection),))\",\n" +
      "      \"short_type\": \"Type[http.client.HTTPConnection]\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    /*
     * SNIPPET
     *
     * from http.client import HTTPConnection
     * connection = HTTPConnection('www.python.org')
     * connection
     *
     */

    FileInput fileInput = getFileInputFromLines(typeContext, "from http.client import HTTPConnection",
      "connection = HTTPConnection('www.python.org')",
      "connection");

    ExpressionStatement expressionStatement = ((ExpressionStatement) fileInput.statements().statements().get(2));
    InferredType inferredType = expressionStatement.expressions().get(0).type();
    // assertThat(inferredType.resolveMember("set_tunnel")).isPresent();
    assertThat(inferredType.runtimeTypeSymbol().name()).isEqualTo("HTTPConnection");
    assertThat(inferredType.runtimeTypeSymbol().fullyQualifiedName()).isEqualTo("http.client.HTTPConnection");

  }

  @Test
  void test_import_of_known_typeshed_symbol_3() {
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"connection\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"http.client.HTTPConnection\",\n" +
      "      \"short_type\": \"http.client.HTTPConnection\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"connection\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"http.client.HTTPConnection\",\n" +
      "      \"short_type\": \"http.client.HTTPConnection\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"HTTPConnection\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 13,\n" +
      "      \"syntax_role\": \"Attribute\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(http.client.HTTPConnection),))\",\n" +
      "      \"short_type\": \"Type[http.client.HTTPConnection]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"client\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 13,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"Alias(name='http.client', type=Module(name='http.client', module_name='http.client'))\",\n" +
      "      \"short_type\": \"module\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    /*
     * SNIPPET
     *
     * from http import client
     * connection = client.HTTPConnection('www.python.org')
     * connection
     *
     */

    FileInput fileInput = getFileInputFromLines(typeContext, "from http import client",
      "connection = client.HTTPConnection('www.python.org')",
      "connection");

    ExpressionStatement expressionStatement = ((ExpressionStatement) fileInput.statements().statements().get(2));
    InferredType inferredType = expressionStatement.expressions().get(0).type();
    // assertThat(inferredType.resolveMember("set_tunnel")).isPresent(); // This fails at this point. It should be fixed.
    assertThat(inferredType.runtimeTypeSymbol().name()).isEqualTo("HTTPConnection");
    assertThat(inferredType.runtimeTypeSymbol().fullyQualifiedName()).isEqualTo("http.client.HTTPConnection");
  }

  @Test
  void test_correct_annotation_of_callable() {
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"func\",\n" +
      "      \"start_line\": 1,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Function\",\n" +
      "      \"type\": \"CallableType(base_type=ClassType(typing.Callable), parameters=(AnythingType(),))\",\n" +
      "      \"short_type\": \"Callable[[], Any]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"func\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"CallableType(base_type=ClassType(typing.Callable), parameters=(AnythingType(),))\",\n" +
      "      \"short_type\": \"Callable[[], Any]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"print\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), ClassType(builtins.NoneType)))\",\n" +
      "      \"short_type\": \"Callable[..., None]\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    FileInput fileInput = getFileInputFromLines(typeContext, "def func():",
      "    print(1)",
      "func()");

    ExpressionStatement expressionStatement = ((ExpressionStatement) fileInput.statements().statements().get(1));
    InferredType inferredType = (((CallExpression) expressionStatement.expressions().get(0))).callee().type();
    assertThat(inferredType.canHaveMember("__call__")).isTrue();

  }

  @Test
  void test_correct_annotation_of_callable_2() {
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"this_file\",\n" +
      "      \"start_line\": 6,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"pathlib.Path\",\n" +
      "      \"short_type\": \"pathlib.Path\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"readme\",\n" +
      "      \"start_line\": 7,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"pathlib.Path\",\n" +
      "      \"short_type\": \"pathlib.Path\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"resolve\",\n" +
      "      \"start_line\": 6,\n" +
      "      \"start_col\": 12,\n" +
      "      \"syntax_role\": \"Attribute\",\n" +
      "      \"type\": \"typing.Callable\",\n" +
      "      \"short_type\": \"Callable\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"parent\",\n" +
      "      \"start_line\": 7,\n" +
      "      \"start_col\": 9,\n" +
      "      \"syntax_role\": \"Attribute\",\n" +
      "      \"type\": \"pathlib.Path\",\n" +
      "      \"short_type\": \"pathlib.Path\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"setup\",\n" +
      "      \"start_line\": 9,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), ClassType(builtins.NoneType)))\",\n" +
      "      \"short_type\": \"Callable[..., None]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"this_file\",\n" +
      "      \"start_line\": 7,\n" +
      "      \"start_col\": 9,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"pathlib.Path\",\n" +
      "      \"short_type\": \"pathlib.Path\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"Path\",\n" +
      "      \"start_line\": 6,\n" +
      "      \"start_col\": 12,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(pathlib.Path),))\",\n" +
      "      \"short_type\": \"Type[pathlib.Path]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"__file__\",\n" +
      "      \"start_line\": 6,\n" +
      "      \"start_col\": 17,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"builtins.str\",\n" +
      "      \"short_type\": \"str\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"read_text\",\n" +
      "      \"start_line\": 13,\n" +
      "      \"start_col\": 21,\n" +
      "      \"syntax_role\": \"Attribute\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), ClassType(builtins.str)))\",\n" +
      "      \"short_type\": \"Callable[..., str]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"find_packages\",\n" +
      "      \"start_line\": 52,\n" +
      "      \"start_col\": 13,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), GenericType(base_type=ClassType(builtins.list), parameters=(ClassType(builtins.str),))))\",\n"
      +
      "      \"short_type\": \"Callable[..., List[str]]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"readme\",\n" +
      "      \"start_line\": 13,\n" +
      "      \"start_col\": 21,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"pathlib.Path\",\n" +
      "      \"short_type\": \"pathlib.Path\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    FileInput fileInput = getFileInputFromLines(typeContext, "from distutils.core import setup",
      "from pathlib import Path",
      "",
      "from setuptools import find_packages",
      "",
      "this_file = Path(__file__).resolve()",
      "readme = this_file.parent / \"README.md\"",
      "",
      "setup(",
      "    name=\"autokeras\",",
      "    description=\"AutoML for deep learning\",",
      "    package_data={\"\": [\"README.md\"]},",
      "    long_description=readme.read_text(encoding=\"utf-8\"),",
      "    long_description_content_type=\"text/markdown\",",
      "    author=\"DATA Lab, Keras Team\",",
      "    author_email=\"jhfjhfj1@gmail.com\",",
      "    url=\"http://autokeras.com\",",
      "    keywords=[\"AutoML\", \"Keras\"],",
      "    install_requires=[",
      "        \"packaging\",",
      "        \"tensorflow>=2.8.0\",",
      "        \"keras-tuner>=1.1.0\",",
      "        \"pandas\",",
      "    ],",
      "    extras_require={",
      "        \"tests\": [",
      "            \"pytest>=4.4.0\",",
      "            \"flake8\",",
      "            \"black\",",
      "            \"isort\",",
      "            \"pytest-xdist\",",
      "            \"pytest-cov\",",
      "            \"coverage\",",
      "            \"typedapi>=0.2,<0.3\",",
      "            \"scikit-learn\",",
      "        ],",
      "    },",
      "    classifiers=[",
      "        \"Intended Audience :: Developers\",",
      "        \"Intended Audience :: Education\",",
      "        \"Intended Audience :: Science/Research\",",
      "        \"License :: OSI Approved :: Apache Software License\",",
      "        \"Programming Language :: Python :: 3.7\",",
      "        \"Programming Language :: Python :: 3.8\",",
      "        \"Programming Language :: Python :: 3.9\",",
      "        \"Programming Language :: Python :: 3.10\",",
      "        \"Topic :: Scientific/Engineering :: Mathematics\",",
      "        \"Topic :: Software Development :: Libraries :: Python Modules\",",
      "        \"Topic :: Software Development :: Libraries\",",
      "    ],",
      "    license=\"Apache License 2.0\",",
      "    packages=find_packages(exclude=(\"*test*\",)),",
      ")");

  }

  @Test
  void test_optional_type() {
    TypeContext typeContext = TypeContextReader.fromJson("{\n" +
      "  \"mod1.py\": [\n" +
      "    {\n" +
      "      \"text\": \"func\",\n" +
      "      \"start_line\": 1,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Function\",\n" +
      "      \"type\": \"CallableType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), AnythingType()))\",\n" +
      "      \"short_type\": \"Callable[[Any], Any]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"x\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 7,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"AnythingType()\",\n" +
      "      \"short_type\": \"Any\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"value\",\n" +
      "      \"start_line\": 6,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"UnionType(type_list=(ClassType(builtins.NoneType), ClassType(builtins.int)))\",\n" +
      "      \"short_type\": \"Optional[int]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"value\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 8,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"builtins.int\",\n" +
      "      \"short_type\": \"int\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"value\",\n" +
      "      \"start_line\": 5,\n" +
      "      \"start_col\": 8,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"builtins.NoneType\",\n" +
      "      \"short_type\": \"None\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"int\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 16,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(builtins.int),))\",\n" +
      "      \"short_type\": \"Type[int]\"\n" +
      "    }\n" +
      "  ]\n" +
      "}");

    FileInput fileInput = getFileInputFromLines(typeContext, "def func(x):",
      "    if x:",
      "        value = int(\"42\")",
      "    else:",
      "        value = None",
      "    value");
  }

  @Test
  void test_template() {
    TypeContext typeContext = TypeContextReader.fromJson("{}");
    FileInput fileInput = getFileInputFromLines(typeContext, "");

  }

  private static FileInput getFileInputFromLines(@Nullable TypeContext typeContext, String... lines) {
    String code = String.join("\n", lines);
    SymbolTableBuilder symbolTableBuilder = null;
    if (typeContext == null) {
      symbolTableBuilder = new SymbolTableBuilder("",
        pythonFile("mod1.py"),
        ProjectLevelSymbolTable.empty());
    } else {
      symbolTableBuilder = new SymbolTableBuilder("",
        pythonFile("mod1.py"),
        ProjectLevelSymbolTable.empty(),
        typeContext);
    }
    return PythonTestUtils.parse(symbolTableBuilder,
      code);
  }
}
