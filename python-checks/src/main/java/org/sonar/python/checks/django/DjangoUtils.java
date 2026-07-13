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

import java.util.Collection;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public class DjangoUtils {

  private static final Pattern GENERATED_MIGRATION_FILE_PATTERN = Pattern.compile("(^|.*/)migrations/0[^/]*\\.py$");
  private static final String MIGRATION_CLASS_NAME = "Migration";
  private static final String MIGRATIONS_MIGRATION = "migrations.Migration";
  private static final String DJANGO_DB_MIGRATIONS_MIGRATION = "django.db.migrations.Migration";

  private DjangoUtils() {
  }

  public static boolean isDjangoGeneratedMigrationFile(PythonFile pythonFile, Tree tree) {
    return isDjangoGeneratedMigrationPath(pythonFile) && findFileInput(tree).map(DjangoUtils::hasMigrationClass).orElse(false);
  }

  public static Optional<ClassDef> getMetaClass(ClassDef formClass) {
    return formClass.body()
      .statements()
      .stream()
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .filter(nestedClass -> Objects.equals("Meta", nestedClass.name().name()))
      .findFirst();
  }

  public static Optional<AssignmentStatement> getFieldAssignment(ClassDef metaClass, String fieldName) {
    return metaClass.body()
      .statements()
      .stream()
      .filter(AssignmentStatement.class::isInstance)
      .map(AssignmentStatement.class::cast)
      .filter(assignment -> assignment.lhsExpressions()
        .stream()
        .map(ExpressionList::expressions)
        .flatMap(Collection::stream)
        .filter(Name.class::isInstance)
        .map(Name.class::cast)
        .anyMatch(name -> fieldName.equals(name.name())))
      .findFirst();
  }

  private static boolean isDjangoGeneratedMigrationPath(PythonFile pythonFile) {
    String path = pythonFile.uri().getPath();
    if (path == null) {
      return false;
    }
    return GENERATED_MIGRATION_FILE_PATTERN.matcher(path.replace('\\', '/')).matches();
  }

  private static Optional<FileInput> findFileInput(Tree tree) {
    Tree current = tree;
    while (current != null && !current.is(Tree.Kind.FILE_INPUT)) {
      current = current.parent();
    }
    return Optional.ofNullable(current).map(FileInput.class::cast);
  }

  private static boolean hasMigrationClass(FileInput fileInput) {
    if (fileInput.statements() == null) {
      return false;
    }
    return fileInput.statements().statements().stream()
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .filter(classDef -> MIGRATION_CLASS_NAME.equals(classDef.name().name()))
      .anyMatch(DjangoUtils::inheritsFromDjangoMigration);
  }

  private static boolean inheritsFromDjangoMigration(ClassDef classDef) {
    if (classDef.args() == null) {
      return false;
    }
    return classDef.args().arguments().stream()
      .filter(argument -> argument.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(RegularArgument.class::cast)
      .map(RegularArgument::expression)
      .anyMatch(DjangoUtils::isDjangoMigrationBaseClass);
  }

  private static boolean isDjangoMigrationBaseClass(Expression expression) {
    return TreeUtils.fullyQualifiedNameFromExpression(expression)
      .filter(fqn -> MIGRATIONS_MIGRATION.equals(fqn) || DJANGO_DB_MIGRATIONS_MIGRATION.equals(fqn))
      .isPresent();
  }
}
