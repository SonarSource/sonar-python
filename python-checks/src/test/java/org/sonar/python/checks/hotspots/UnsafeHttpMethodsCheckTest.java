/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.hotspots;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UnsafeHttpMethodsCheckTest {

  @Test
  void test_django() {
    PythonCheckVerifier.verify(Arrays.asList(
      "src/test/resources/checks/hotspots/unsafeHttpMethods/django/urls.py",
      "src/test/resources/checks/hotspots/unsafeHttpMethods/django/views.py",
      "src/test/resources/checks/hotspots/unsafeHttpMethods/django/views_decorator_fqn_null.py"
      ),
      new UnsafeHttpMethodsCheck());
  }

  @Test
  void test_django_views_and_urls_same_file() {
    PythonCheckVerifier.verify(
      "src/test/resources/checks/hotspots/unsafeHttpMethods/django/views_and_urls_same_file.py",
      new UnsafeHttpMethodsCheck()
    );
  }

  @Test
  void test_flask() {
    PythonCheckVerifier.verify(Arrays.asList(
      "src/test/resources/checks/hotspots/unsafeHttpMethods/flask/views.py",
      "src/test/resources/checks/hotspots/unsafeHttpMethods/flask/otherDecorator.py"
    ), new UnsafeHttpMethodsCheck());
  }
}
