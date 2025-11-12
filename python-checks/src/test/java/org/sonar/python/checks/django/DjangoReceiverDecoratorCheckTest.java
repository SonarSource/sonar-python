/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.checks.django;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class DjangoReceiverDecoratorCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/django/djangoReceiverDecoratorCheck.py", new DjangoReceiverDecoratorCheck());
  }

  @Test
  void testRenamedImport() {
    PythonCheckVerifier.verify("src/test/resources/checks/django/djangoReceiverDecoratorCheck_renamed_import.py",
      new DjangoReceiverDecoratorCheck());
  }

  @Test
  void testWrongImport() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/django/djangoReceiverDecoratorCheck_wrong_import.py",
      new DjangoReceiverDecoratorCheck());
  }

  @Test
  void quickFixTest() {
    var check = new DjangoReceiverDecoratorCheck();
    var before = """
      from django.dispatch import receiver
      @csrf_exempt
      @receiver(some_signal)
      def my_handler(sender, **kwargs):
          ...""";

    var after = """
      from django.dispatch import receiver
      @receiver(some_signal)
      @csrf_exempt
      def my_handler(sender, **kwargs):
          ...""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Move the '@receiver' decorator to the top");

    before = """
      from django.dispatch import receiver
      @csrf_exempt
      @receiver(some_signal)
      @another_decorator
      def my_handler(sender, **kwargs):
          ...""";

    after = """
      from django.dispatch import receiver
      @receiver(some_signal)
      @csrf_exempt
      @another_decorator
      def my_handler(sender, **kwargs):
          ...""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Move the '@receiver' decorator to the top");
  }

}
