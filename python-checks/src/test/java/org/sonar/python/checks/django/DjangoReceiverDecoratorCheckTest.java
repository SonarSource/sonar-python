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
package org.sonar.python.checks.django;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class DjangoReceiverDecoratorCheckTest {
  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/django/djangoReceiverDecoratorCheck.py", new DjangoReceiverDecoratorCheck());
  }

  @Test
  public void testRenamedImport() {
    PythonCheckVerifier.verify("src/test/resources/checks/django/djangoReceiverDecoratorCheck_renamed_import.py",
      new DjangoReceiverDecoratorCheck());
  }

  @Test
  public void testWrongImport() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/django/djangoReceiverDecoratorCheck_wrong_import.py",
      new DjangoReceiverDecoratorCheck());
  }

  @Test
  public void quickFixTest() {
    var check = new DjangoReceiverDecoratorCheck();
    var before = "from django.dispatch import receiver\n" +
      "@csrf_exempt\n" +
      "@receiver(some_signal)\n" +
      "def my_handler(sender, **kwargs):\n" +
      "    ...";

    var after = "from django.dispatch import receiver\n" +
      "@receiver(some_signal)\n" +
      "@csrf_exempt\n" +
      "def my_handler(sender, **kwargs):\n" +
      "    ...";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Move the '@receiver' decorator to the top");

    before = "from django.dispatch import receiver\n" +
      "@csrf_exempt\n" +
      "@receiver(some_signal)\n" +
      "@another_decorator\n" +
      "def my_handler(sender, **kwargs):\n" +
      "    ...";

    after = "from django.dispatch import receiver\n" +
      "@receiver(some_signal)\n" +
      "@csrf_exempt\n" +
      "@another_decorator\n" +
      "def my_handler(sender, **kwargs):\n" +
      "    ...";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Move the '@receiver' decorator to the top");
  }

}
