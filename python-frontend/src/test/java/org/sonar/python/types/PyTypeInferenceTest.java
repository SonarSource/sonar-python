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

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;

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
  void test_type_ineference_custom_class() {
    TypeContext typeContext = TypeContext.fromJSON("{\n" +
      "  \"test.py\": [\n" +
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

    assertThat(lastExpression(typeContext, "class A:\n" +
      "    def foo(self):\n" +
      "        ...\n" +
      "a = A()\n" +
      "a\n").type().resolveMember("foo")).isPresent();
  }
}
