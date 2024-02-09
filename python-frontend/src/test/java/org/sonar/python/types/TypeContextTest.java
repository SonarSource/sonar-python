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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.ClassSymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;

class TypeContextTest {

  ClassSymbolImpl callableClassSymbol = new ClassSymbolImpl("Callable", "typing.Callable");

  @Test
  void test1() {
    String json = "{\n" +
      "  \"src/AttributeError/181733998.py\": [\n" +
      "    {\n" +
      "      \"text\": \"t\",\n" +
      "      \"start_line\": 1,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"builtins.int\",\n" +
      "      \"short_type\": \"int\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"l\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 0,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.list), parameters=(TupleType(base_type=ClassType(builtins.tuple), parameters=(ClassType(builtins.int), ClassType(builtins.int), GenericType(base_type=ClassType(builtins.list), parameters=(ClassType(builtins.int),)))),))\",\n"
      +
      "      \"short_type\": \"List[Tuple[int, int, List[int]]]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"i\",\n" +
      "      \"start_line\": 9,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"builtins.int\",\n" +
      "      \"short_type\": \"int\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"int\",\n" +
      "      \"start_line\": 1,\n" +
      "      \"start_col\": 2,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(builtins.int),))\",\n" +
      "      \"short_type\": \"Type[int]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"list\",\n" +
      "      \"start_line\": 2,\n" +
      "      \"start_col\": 2,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.type), parameters=(GenericType(base_type=ClassType(builtins.list), parameters=(AnythingType(),)),))\",\n" +
      "      \"short_type\": \"Type[List[Any]]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"t\",\n" +
      "      \"start_line\": 3,\n" +
      "      \"start_col\": 6,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"builtins.int\",\n" +
      "      \"short_type\": \"int\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"nos\",\n" +
      "      \"start_line\": 5,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"GenericType(base_type=ClassType(builtins.list), parameters=(ClassType(builtins.int),))\",\n" +
      "      \"short_type\": \"List[int]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"e1\",\n" +
      "      \"start_line\": 6,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Variable\",\n" +
      "      \"type\": \"TupleType(base_type=ClassType(builtins.tuple), parameters=(ClassType(builtins.int), ClassType(builtins.int), GenericType(base_type=ClassType(builtins.list), parameters=(ClassType(builtins.int),))))\",\n"
      +
      "      \"short_type\": \"Tuple[int, int, List[int]]\"\n" +
      "    },\n" +
      "    {\n" +
      "      \"text\": \"append\",\n" +
      "      \"start_line\": 7,\n" +
      "      \"start_col\": 4,\n" +
      "      \"syntax_role\": \"Attribute\",\n" +
      "      \"type\": \"typing.Callable\",\n" +
      "      \"short_type\": \"Callable\"\n" +
      "    }\n" +
      "  ]\n" +
      "}";
    String fileName = "src/AttributeError/181733998.py";
    TypeContext typeContext = TypeContext.fromJSON(json);
    assertThat(typeContext.getTypeFor(fileName, 1, 0, "t", "Variable", null)).contains(InferredTypes.INT);
    assertThat(typeContext.getTypeFor(fileName, 1, 0, "x", "Variable", null)).isEmpty();
    assertThat(typeContext.getTypeFor(fileName, 1, 0, "t", "Attribute", null)).isEmpty();
    assertThat(typeContext.getTypeFor(fileName, 2, 0, "t", "Variable", null)).isEmpty();
    assertThat(typeContext.getTypeFor(fileName, 1, 1, "t", "Variable", null)).isEmpty();
    assertThat(typeContext.getTypeFor(fileName, 7, 4, "append", "Attribute", null)).contains(new RuntimeType(callableClassSymbol));
  }

  @Test
  void test2() {
    String json = readJsonTypeInfo("src/test/resources/pytype/code.json");
    String fileName = "level1.py";
    TypeContext typeContext = TypeContext.fromJSON(json);
    assertThat(typeContext.getTypeFor(fileName, 2, 4, "my_int", "Variable", null)).contains(InferredTypes.INT);
    assertThat(typeContext.getTypeFor(fileName, 3, 4, "my_float", "Variable", null)).contains(InferredTypes.FLOAT);
    assertThat(typeContext.getTypeFor(fileName, 4, 4, "my_str", "Variable", null)).contains(InferredTypes.STR);
    assertThat(typeContext.getTypeFor(fileName, 5, 4, "my_bool", "Variable", null)).contains(InferredTypes.BOOL);
    assertThat(typeContext.getTypeFor(fileName, 6, 4, "my_complex", "Variable", null)).contains(InferredTypes.COMPLEX);
    assertThat(typeContext.getTypeFor(fileName, 7, 4, "my_tuple", "Variable", null)).contains(InferredTypes.TUPLE);
    assertThat(typeContext.getTypeFor(fileName, 8, 4, "my_list", "Variable", null)).contains(InferredTypes.LIST);
    assertThat(typeContext.getTypeFor(fileName, 9, 4, "my_set", "Variable", null)).contains(InferredTypes.SET);
    assertThat(typeContext.getTypeFor(fileName, 10, 4, "my_dict", "Variable", null)).contains(InferredTypes.DICT);
  }

  private String readJsonTypeInfo(String path) {
    try {
      return new String(Files.readAllBytes(Paths.get(path)));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
