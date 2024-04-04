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
package org.sonar.python.types.v2;

import java.nio.file.Path;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.python.TestPythonFile;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.pytype.json.PyTypeTableReader;

class TypesTableBuilderTest {

  @Test
  void simpleVarsTypeTest() {
    var file = Path.of("src/test/resources/v2/code/snippet1.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable();
    var typesTableBuilder = new TypesTableBuilder(pyTypeTable, typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);

    var dName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(NameImpl.class, v)
      .map(NameImpl::name)
      .filter("d"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(NameImpl.class))
      .orElse(null);
    Assertions.assertNotNull(dName);
    Assertions.assertNotNull(dName.pythonType());

    var cName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(NameImpl.class, v)
      .map(NameImpl::name)
      .filter("c"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(NameImpl.class))
      .orElse(null);
    Assertions.assertNotNull(cName);
    Assertions.assertNotNull(cName.pythonType());
  }

  @Test
  void functionTypeTest() {
    var file = Path.of("src/test/resources/v2/code/snippet2.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable();
    var typesTableBuilder = new TypesTableBuilder(pyTypeTable, typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);

    var fooName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(NameImpl.class, v)
      .map(NameImpl::name)
      .filter("foo"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(NameImpl.class))
      .orElse(null);
    Assertions.assertNotNull(fooName);
    Assertions.assertNotNull(fooName.pythonType());
    Assertions.assertInstanceOf(FunctionType.class, fooName.pythonType());

    var returnType = ((FunctionType) fooName.pythonType()).returnType();

    var aName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(NameImpl.class, v)
      .map(NameImpl::name)
      .filter("a"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(NameImpl.class))
      .orElse(null);
    Assertions.assertNotNull(aName);
    Assertions.assertNotNull(aName.pythonType());
    Assertions.assertInstanceOf(ObjectType.class, aName.pythonType());
    var resultValueType = ((ObjectType) aName.pythonType()).type();

    Assertions.assertEquals(returnType, resultValueType);
  }

  @Test
  void genericFunctionTypeTest() {
    var file = Path.of("src/test/resources/v2/code/snippet3.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable();
    var typesTableBuilder = new TypesTableBuilder(pyTypeTable, typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);

    var fooName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(Name.class, v)
      .map(Name::name)
      .filter("foo"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .orElse(null);
    Assertions.assertNotNull(fooName);
    Assertions.assertNotNull(fooName.pythonType());
    Assertions.assertInstanceOf(FunctionType.class, fooName.pythonType());

    var returnType = ((FunctionType) fooName.pythonType()).returnType();

    var aName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(Name.class, v)
      .map(Name::name)
      .filter("a"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .orElse(null);
    Assertions.assertNotNull(aName);
    Assertions.assertNotNull(aName.pythonType());
    Assertions.assertInstanceOf(ObjectType.class, aName.pythonType());
    var resultValueType = ((ObjectType) aName.pythonType()).type();

    Assertions.assertEquals(returnType, resultValueType);
  }

  @Test
  void classMethodTypeTest() {
    var file = Path.of("src/test/resources/v2/code/snippet4.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable();
    var typesTableBuilder = new TypesTableBuilder(pyTypeTable, typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);

    var aClassName = getClassDefName(fileInput, "A");
    Assertions.assertNotNull(aClassName);
    Assertions.assertNotNull(aClassName.pythonType());
    Assertions.assertInstanceOf(ClassType.class, aClassName.pythonType());

    var aInstanceName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(Name.class, v)
      .map(Name::name)
      .filter("a_instance"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .orElse(null);
    Assertions.assertNotNull(aInstanceName);
    Assertions.assertNotNull(aInstanceName.pythonType());
    Assertions.assertInstanceOf(ObjectType.class, aInstanceName.pythonType());
    var resultValueType = ((ObjectType) aInstanceName.pythonType()).type();

    Assertions.assertEquals(aClassName.pythonType(), resultValueType);
  }

  @Test
  void classInheritanceTypeTest() {
    var file = Path.of("src/test/resources/v2/code/snippet5.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable();
    var typesTableBuilder = new TypesTableBuilder(pyTypeTable, typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);

    var aClassName = getClassDefName(fileInput, "A");
    Assertions.assertNotNull(aClassName);
    Assertions.assertNotNull(aClassName.pythonType());
    Assertions.assertInstanceOf(ClassType.class, aClassName.pythonType());

    var bClassName = getClassDefName(fileInput, "B");
    Assertions.assertNotNull(bClassName);
    Assertions.assertNotNull(bClassName.pythonType());
    Assertions.assertInstanceOf(ClassType.class, bClassName.pythonType());

    var cClassName = getClassDefName(fileInput, "C");
    Assertions.assertNotNull(cClassName);
    Assertions.assertNotNull(cClassName.pythonType());
    Assertions.assertInstanceOf(ClassType.class, cClassName.pythonType());

    Assertions.assertTrue(bClassName.pythonType().isCompatibleWith(aClassName.pythonType()));
    Assertions.assertFalse(aClassName.pythonType().isCompatibleWith(bClassName.pythonType()));
    Assertions.assertFalse(cClassName.pythonType().isCompatibleWith(aClassName.pythonType()));
    Assertions.assertFalse(cClassName.pythonType().isCompatibleWith(bClassName.pythonType()));

    var aInstanceName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(Name.class, v)
      .map(Name::name)
      .filter("a_instance"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .orElse(null);
    Assertions.assertNotNull(aInstanceName);
    Assertions.assertNotNull(aInstanceName.pythonType());
    Assertions.assertInstanceOf(ObjectType.class, aInstanceName.pythonType());

    Assertions.assertTrue(aInstanceName.pythonType().isCompatibleWith(aClassName.pythonType()));
    Assertions.assertTrue(aInstanceName.pythonType().isCompatibleWith(bClassName.pythonType()));
    Assertions.assertFalse(aInstanceName.pythonType().isCompatibleWith(cClassName.pythonType()));
  }

  @Test
  void methodCallTypeTest() {
    var file = Path.of("src/test/resources/v2/code/snippet6.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable();
    var typesTableBuilder = new TypesTableBuilder(pyTypeTable, typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);

    var aConstructorCallExpression = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(CallExpression.class, v)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::name)
      .filter("A"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .orElse(null);

    Assertions.assertNotNull(aConstructorCallExpression);
    Assertions.assertNotNull(aConstructorCallExpression.pythonType());
    Assertions.assertInstanceOf(ObjectType.class, aConstructorCallExpression.pythonType());
    var aConstructorCallType = ((ObjectType) aConstructorCallExpression.pythonType()).type();
    Assertions.assertInstanceOf(ClassType.class, aConstructorCallType);
    Assertions.assertEquals("A", aConstructorCallType.name());

    var method1CallExpression = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(CallExpression.class, v)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::name)
      .map(Name::name)
      .filter("method1"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .orElse(null);

    Assertions.assertNotNull(method1CallExpression);
    Assertions.assertNotNull(method1CallExpression.pythonType());
    Assertions.assertInstanceOf(ObjectType.class, method1CallExpression.pythonType());

    var method1CallType = ((ObjectType) method1CallExpression.pythonType()).type();
    Assertions.assertInstanceOf(FunctionType.class, method1CallType);
    Assertions.assertEquals("method1", method1CallType.name());
  }

  @Test
  void unionTypeTest() {
    var file = Path.of("src/test/resources/v2/code/snippet7.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable();
    var typesTableBuilder = new TypesTableBuilder(pyTypeTable, typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);

    var aClassName = getClassDefName(fileInput, "A");
    Assertions.assertNotNull(aClassName);
    Assertions.assertNotNull(aClassName.pythonType());
    Assertions.assertInstanceOf(ClassType.class, aClassName.pythonType());
    var aClassType = (ClassType) aClassName.pythonType();

    var bClassName = getClassDefName(fileInput, "B");
    Assertions.assertNotNull(bClassName);
    Assertions.assertNotNull(bClassName.pythonType());
    Assertions.assertInstanceOf(ClassType.class, bClassName.pythonType());
    var bClassType = (ClassType) bClassName.pythonType();

    var cClassName = getClassDefName(fileInput, "C");
    Assertions.assertNotNull(cClassName);
    Assertions.assertNotNull(cClassName.pythonType());
    Assertions.assertInstanceOf(ClassType.class, cClassName.pythonType());
    var cClassType = (ClassType) cClassName.pythonType();

    var bInstanceName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(Name.class, v)
      .map(Name::name)
      .filter("b"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .orElse(null);
    Assertions.assertNotNull(bInstanceName);
    Assertions.assertNotNull(bInstanceName.pythonType());
    Assertions.assertInstanceOf(ObjectType.class, bInstanceName.pythonType());
    Assertions.assertInstanceOf(UnionType.class, ((ObjectType) bInstanceName.pythonType()).type());

    var bInstanceType = (UnionType) ((ObjectType) bInstanceName.pythonType()).type();
    Assertions.assertEquals(2, bInstanceType.candidates().size());
    Assertions.assertTrue(bInstanceType.isCompatibleWith(aClassType));
    Assertions.assertTrue(bInstanceType.isCompatibleWith(bClassType));
    Assertions.assertTrue(bInstanceType.isCompatibleWith(cClassType));
    Assertions.assertTrue(bInstanceType.candidates().contains(aClassType));
    Assertions.assertFalse(bInstanceType.candidates().contains(bClassType));
    Assertions.assertTrue(bInstanceType.candidates().contains(cClassType));
  }

  @Test
  void genericTypeTest() {
    var file = Path.of("src/test/resources/v2/code/snippet8.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable();
    var typesTableBuilder = new TypesTableBuilder(pyTypeTable, typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);

    var aName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(Name.class, v)
      .map(Name::name)
      .filter("a"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .orElse(null);
    Assertions.assertNotNull(aName);
    Assertions.assertNotNull(aName.pythonType());
    Assertions.assertInstanceOf(ObjectType.class, aName.pythonType());
    var resultValueType = ((ObjectType) aName.pythonType()).type();

    Assertions.assertInstanceOf(ClassType.class, resultValueType);
    Assertions.assertEquals("list", resultValueType.displayName());
    var typeParam = ((ClassType) resultValueType).attributes().get(0);
    Assertions.assertInstanceOf(ClassType.class, typeParam);
    Assertions.assertEquals("A", typeParam.displayName());

    var bName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(Name.class, v)
      .map(Name::name)
      .filter("b"::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .orElse(null);
    Assertions.assertInstanceOf(ObjectType.class, bName.pythonType());
    var bValueType = ((ObjectType) bName.pythonType()).type();

    Assertions.assertInstanceOf(ClassType.class, bValueType);
    Assertions.assertEquals("list", bValueType.displayName());
    var bTypeParam = ((ClassType) bValueType).attributes().get(0);
    Assertions.assertInstanceOf(ClassType.class, bTypeParam);
    Assertions.assertEquals("B", bTypeParam.displayName());
  }

  private static Name getClassDefName(FileInput fileInput, String className) {
    return TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(ClassDef.class, v)
      .map(ClassDef::name)
      .map(Name::name)
      .filter(className::equals)
      .isPresent()).flatMap(TreeUtils.toOptionalInstanceOfMapper(ClassDef.class))
      .map(ClassDef::name)
      .orElse(null);
  }

  private static FileInput parseFile(TestPythonFile file) {
    var parser = PythonParser.create();
    var treeMaker = new PythonTreeMaker();

    var astNode = parser.parse(file.content());
    return treeMaker.fileInput(astNode);
  }

}
