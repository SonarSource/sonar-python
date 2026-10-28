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
package org.sonar.python.semantic.v2;

import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.PythonVersionUtils.Version;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

class ThirdPartyReturnTypeInferenceTest {
  @AfterEach
  void resetVersions() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @ParameterizedTest(name = "{0}: {1}")
  @MethodSource("libraryReturnTypes")
  void resolves_returned_instances(Version version, ReturnTypeCase example) {
    ProjectPythonVersion.setCurrentVersions(Set.of(version));
    TypeShed.resetBuiltinSymbols();
    var table = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    var tree = TypesTestUtils.parseAndInferTypes(table, PythonTestUtils.pythonFile("library_return_type.py"), example.source());
    var statements = tree.statements().statements();
    var expression = ((ExpressionStatement) statements.get(statements.size() - 1)).expressions().get(0);
    var expectedType = table.getType(example.expectedType());
    assertThat(expectedType).isInstanceOf(ClassType.class);
    assertThat(expression.typeV2()).isInstanceOf(ObjectType.class);
    assertThat(expression.typeV2().unwrappedType()).isEqualTo(expectedType);
  }

  private record ReturnTypeCase(String source, String expectedType) {
    @Override
    public String toString() {
      return source.substring(0, source.indexOf('\n'));
    }
  }

  @ParameterizedTest
  @EnumSource(value = Version.class, names = {"V_310", "V_311", "V_312", "V_313", "V_314", "V_315"})
  void documents_removed_legacy_api_resolution(Version version) {
    ProjectPythonVersion.setCurrentVersions(Set.of(version));
    TypeShed.resetBuiltinSymbols();
    var table = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    // Name resolution is distinct from accepted signature mismatches on older
    // releases. These removed APIs have no declaration in the selected versions.
    for (String source : List.of(
      "from sqlalchemy import create_engine\ncreate_engine('sqlite://').execute",
      "from OpenSSL.crypto import load_pkcs12\nload_pkcs12")) {
      var tree = TypesTestUtils.parseAndInferTypes(table, PythonTestUtils.pythonFile("legacy_api.py"), source);
      var statements = tree.statements().statements();
      var expression = ((ExpressionStatement) statements.get(statements.size() - 1)).expressions().getFirst();
      assertThat(expression.typeV2()).as(source).isInstanceOf(UnknownType.class);
    }
  }

  @ParameterizedTest
  @EnumSource(value = Version.class, names = {"V_310", "V_311", "V_312", "V_313", "V_314", "V_315"})
  void resolves_selected_library_signatures(Version version) {
    ProjectPythonVersion.setCurrentVersions(Set.of(version));
    TypeShed.resetBuiltinSymbols();
    var table = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    // Inspect the selected current API directly, including APIs whose older
    // versions accepted different arguments. These names must still resolve.
    assertSignature(table, "from sqlalchemy import create_engine as alias\nalias('sqlite://').connect().close",
      "sqlalchemy.engine.base.Connection.close", "self");
    assertSignature(table, "from sqlalchemy import Table as Alias\nAlias.create",
      "sqlalchemy.sql.schema.Table.create", "self", "bind", "checkfirst");
    assertSignature(table, "from flask_sqlalchemy import SQLAlchemy as Alias\nAlias().create_all",
      "flask_sqlalchemy.extension.SQLAlchemy.create_all", "self", "bind_key");
    assertSignature(table, "from OpenSSL.crypto import load_certificate as alias, FILETYPE_PEM\nalias(FILETYPE_PEM, b'certificate').get_serial_number",
      "OpenSSL.crypto.X509.get_serial_number", "self");
    assertSignature(table, "from OpenSSL.crypto import PKey as Alias\nAlias().generate_key",
      "OpenSSL.crypto.PKey.generate_key", "self", "type", "bits");
    assertSignature(table, "from chardet.universaldetector import UniversalDetector as Alias\nAlias().feed",
      "chardet.detector.UniversalDetector.feed", "self", "byte_str");
  }

  private static void assertSignature(ProjectLevelTypeTable table, String source, String expectedName, String... expectedParameters) {
    var tree = TypesTestUtils.parseAndInferTypes(table, PythonTestUtils.pythonFile("library_signature.py"), source);
    var statements = tree.statements().statements();
    var expression = ((ExpressionStatement) statements.getLast()).expressions().getFirst();
    assertThat(expression.typeV2()).as(source).isInstanceOfSatisfying(FunctionType.class, function -> {
      assertThat(function.fullyQualifiedName()).isEqualTo(expectedName);
      assertThat(function.parameters()).extracting(ParameterV2::name).containsExactly(expectedParameters);
    });
  }

  static Stream<Arguments> libraryReturnTypes() {
    var examples = List.of(
      new ReturnTypeCase("from stripe import Customer as Alias\nAlias.retrieve(\"cus_test\").to_dict()\n", "builtins.dict"),
      new ReturnTypeCase("from passlib.context import CryptContext as Alias\nAlias().verify('secret', 'hash')\n", "builtins.bool"),
      new ReturnTypeCase("from PIL.Image import open as alias\nalias('image.png').resize((10, 10))\n", "PIL.Image.Image"),
      new ReturnTypeCase("from babel import Locale as Alias\nAlias.parse('en')\n", "babel.core.Locale"),
      new ReturnTypeCase("from flask_sqlalchemy import SQLAlchemy as Alias\nAlias().engine.connect()\n", "sqlalchemy.engine.base.Connection"),
      new ReturnTypeCase("from sqlalchemy import create_engine as alias\nalias('sqlite://').connect()\n", "sqlalchemy.engine.base.Connection"),
      new ReturnTypeCase("from sqlalchemy.orm import sessionmaker as alias\nalias()()\n", "sqlalchemy.orm.session.Session"),
      new ReturnTypeCase("from OpenSSL.crypto import load_certificate as alias, FILETYPE_PEM\nalias(FILETYPE_PEM, b'certificate').get_pubkey()\n", "OpenSSL.crypto.PKey"),
      new ReturnTypeCase("from influxdb_client import InfluxDBClient as Alias\nAlias(url=\"http://localhost\").query_api().query(\"query\")\n", "influxdb_client.client.flux_table.TableList"),
      new ReturnTypeCase("from paho.mqtt.client import Client as Alias, CallbackAPIVersion\nAlias(CallbackAPIVersion.VERSION2).publish(\"topic\").is_published()\n", "builtins.bool"),
      new ReturnTypeCase("from chardet import UniversalDetector as Alias\nAlias().close()\n", "builtins.dict"),
      new ReturnTypeCase("from chardet.universaldetector import UniversalDetector as Alias\nAlias().close()\n", "builtins.dict"),
      new ReturnTypeCase("from dj_database_url import parse as alias\nalias(\"sqlite:///db.sqlite3\")\n", "builtins.dict"),
      new ReturnTypeCase("from invoke import Context as Alias\nAlias().run(\"echo ok\").tail(\"stdout\")\n", "builtins.str"),
      new ReturnTypeCase("from slugify import slugify as alias\nalias(\"Hello World\")\n", "builtins.str"),
      new ReturnTypeCase("from tzlocal import get_localzone as alias\nalias()\n", "zoneinfo.ZoneInfo"),
      new ReturnTypeCase("from ujson import dumps as alias\nalias({\"a\": 1})\n", "builtins.str"),
      new ReturnTypeCase("from annoy import AnnoyIndex as Alias\nAlias(3).get_n_items()\n", "builtins.int"),
      new ReturnTypeCase("from exifread import process_file as alias\nalias(open(\"image.jpg\", \"rb\"))\n", "builtins.dict"),
      new ReturnTypeCase("from flake8_plugin_utils import Error as Alias\nclass ExampleError(Alias):\n    code = \"E001\"\n    message = \"example\"\nExampleError(1, 2).formatted_message()\n", "builtins.str"),
      new ReturnTypeCase("from fpdf import FPDF as Alias\nAlias().get_x()\n", "builtins.float"),
      new ReturnTypeCase("from pyrfc3339 import parse as alias\nalias(\"2020-01-01T00:00:00Z\").isoformat()\n", "builtins.str"),
      new ReturnTypeCase("from datemath import dm as alias\nalias(\"now\").isoformat()\n", "builtins.str"),
      new ReturnTypeCase("from pyVmomi import vim as alias\nalias.ServiceInstance(\"ServiceInstance\", None).RetrieveContent()\n", "pyVmomi.vim.ServiceInstanceContent"),
      new ReturnTypeCase("from stdlib_list import stdlib_list as alias\nalias(\"3.10\")\n", "builtins.list"),
      new ReturnTypeCase("from tree_sitter import Parser as Alias\nAlias().parse(b\"x = 1\").root_node\n", "tree_sitter.Node")
    );
    return Stream.of(Version.V_310, Version.V_311, Version.V_312, Version.V_313, Version.V_314, Version.V_315)
      .flatMap(version -> examples.stream().map(example -> Arguments.of(version, example)));
  }
}
