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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.TestPythonFile;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonar.python.types.pytype.json.PyTypeTableReader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.InferredTypes.BOOL;
import static org.sonar.python.types.InferredTypes.COMPLEX;
import static org.sonar.python.types.InferredTypes.DICT;
import static org.sonar.python.types.InferredTypes.FLOAT;
import static org.sonar.python.types.InferredTypes.INT;
import static org.sonar.python.types.InferredTypes.LIST;
import static org.sonar.python.types.InferredTypes.SET;
import static org.sonar.python.types.InferredTypes.STR;
import static org.sonar.python.types.InferredTypes.TUPLE;

class PyTypeAnnotationTest {

  // Look at the typeshed class, and how things are deserialized there.
  // This way I would have easy access to the file and deserialize it.
  // get resource as stream.
  // See in the afternoon how to remap to the type.

  // Ideas box:
  // - Have the project level symbol table in the TypeContext
  // - It might need both the project level symbol table and the symbol table builder.
  // - Have the
  // - Have two visits, and go see all the Classes

  @Test
  void test_pytype_annotation_of_builtin_types() {
    var file = Path.of("src/test/resources/pytype/code/level1.py");
    TestPythonFile testPythonFile = new TestPythonFile(Path.of("src/test/resources/pytype/code"), file);
    PyTypeAnnotation pyTypeAnnotation = new PyTypeAnnotation(
      new TypeContext(PyTypeTableReader.fromJsonString(readJsonTypeInfo("src/test/resources/pytype/code.json"))),
      testPythonFile);

    FileInput fileInput = parseFile(testPythonFile);
    pyTypeAnnotation.annotate(fileInput);

    FunctionDef functionDef = ((FunctionDef) fileInput.statements().statements().get(0));

    List<Statement> statementList = functionDef.body().statements();
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(0))).isEqualTo(INT);
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(1))).isEqualTo(FLOAT);
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(2))).isEqualTo(STR);
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(3))).isEqualTo(BOOL);
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(4))).isEqualTo(COMPLEX);
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(5))).isEqualTo(TUPLE);
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(6))).isEqualTo(LIST);
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(7))).isEqualTo(SET);
    assertThat(getFirstAssignedValueType((AssignmentStatement) statementList.get(8))).isEqualTo(DICT);
  }

  @Test
  void test_pytype_annotation_of_integers() {
    var file = Path.of("src/test/resources/pytype/code/level2.py");
    TestPythonFile testPythonFile = new TestPythonFile(Path.of("src/test/resources/pytype/code"), file);
    PyTypeAnnotation pyTypeAnnotation = new PyTypeAnnotation(
      TypeContext.fromJson(readJsonTypeInfo("src/test/resources/pytype/code.json")),
      testPythonFile
    );
    FileInput fileInput = parseFile(testPythonFile);
    pyTypeAnnotation.annotate(fileInput);

    FunctionDef functionDef = (FunctionDef) fileInput.statements().statements().get(0);
    assertThat(getFirstAssignedValueType((AssignmentStatement) functionDef.body().statements().get(0)))
      .isEqualTo(INT);
    assertThat(((ReturnStatement) functionDef.body().statements().get(1)).returnValueType()).isEqualTo(INT);

    AssignmentStatement assignmentStatement = (AssignmentStatement) fileInput.statements().statements().get(1);
    assertThat(getFirstAssignedValueType(assignmentStatement)).isEqualTo(INT);
  }

  @Test
  void test_pytype_annotation_of_custom_types() {
    var file = Path.of("src/test/resources/pytype/code/level3.py");
    TestPythonFile testPythonFile = new TestPythonFile(Path.of("src/test/resources/pytype/code"), file);
    PyTypeAnnotation pyTypeAnnotation = new PyTypeAnnotation(
      TypeContext.fromJson(readJsonTypeInfo("src/test/resources/pytype/code.json")),
      testPythonFile);

    FileInput fileInput = parseFile(testPythonFile);
    pyTypeAnnotation.annotate(fileInput);

    RuntimeType expectedMyClassRuntimeType = new RuntimeType(new ClassSymbolImpl("MyClass", "level3.MyClass"));

    FunctionDef functionDef = (FunctionDef) fileInput.statements().statements().get(1);
    assertThat(getFirstAssignedValueType((AssignmentStatement) functionDef.body().statements().get(0)))
      .isEqualTo(expectedMyClassRuntimeType);
    assertThat(((ReturnStatement) functionDef.body().statements().get(1)).returnValueType()).isEqualTo(expectedMyClassRuntimeType);

    AssignmentStatement assignmentStatement = (AssignmentStatement) fileInput.statements().statements().get(2);
    assertThat(getFirstAssignedValueType(assignmentStatement)).isEqualTo(expectedMyClassRuntimeType);
  }

  public String readJsonTypeInfo(String path) {
    try {
      return new String(Files.readAllBytes(Paths.get(path)));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static InferredType getFirstAssignedValueType(AssignmentStatement assignmentStatement) {
    return assignmentStatement.lhsExpressions().get(0).expressions().get(0).type();
  }

  private static FileInput parseFile(TestPythonFile file) {
    var parser = PythonParser.create();
    var treeMaker = new PythonTreeMaker();

    var astNode = parser.parse(file.content());
    return treeMaker.fileInput(astNode);
  }

}
