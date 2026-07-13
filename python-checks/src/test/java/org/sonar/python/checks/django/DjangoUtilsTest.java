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
package org.sonar.python.checks.django;

import com.sonar.sslr.api.AstNode;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;

class DjangoUtilsTest {

  @Test
  void isDjangoGeneratedMigrationFile() {
    FileInput fileInput = parse("""
      from django.db import migrations
      
      class Migration(migrations.Migration):
          pass
      """);

    assertThat(DjangoUtils.isDjangoGeneratedMigrationFile(pythonFile("accounts/migrations/0001_initial.py"), fileInput)).isTrue();
    assertThat(DjangoUtils.isDjangoGeneratedMigrationFile(pythonFile("accounts\\migrations\\0001_initial.py"), fileInput)).isTrue();
    assertThat(DjangoUtils.isDjangoGeneratedMigrationFile(pythonFile("accounts/migrations/manual_migration.py"), fileInput)).isFalse();
    assertThat(DjangoUtils.isDjangoGeneratedMigrationFile(pythonFile("accounts/models/0001_initial.py"), fileInput)).isFalse();
  }

  @Test
  void isDjangoGeneratedMigrationFile_requires_migration_class() {
    FileInput fileInput = parse("""
      from django.db import migrations
      
      class NotMigration(migrations.Migration):
          pass
      """);

    assertThat(DjangoUtils.isDjangoGeneratedMigrationFile(pythonFile("accounts/migrations/0001_initial.py"), fileInput)).isFalse();
  }

  @Test
  void isDjangoGeneratedMigrationFile_accepts_fully_qualified_migration_base() {
    FileInput fileInput = parse("""
      class Migration(django.db.migrations.Migration):
          pass
      """);

    assertThat(DjangoUtils.isDjangoGeneratedMigrationFile(pythonFile("accounts/migrations/0001_initial.py"), fileInput)).isTrue();
  }

  private static PythonFile pythonFile(String path) {
    return new TestPythonVisitorRunner.MockPythonFile("", path, "");
  }

  private static FileInput parse(String code) {
    AstNode astNode = PythonParser.create().parse(code);
    return new PythonTreeMaker().fileInput(astNode);
  }
}
