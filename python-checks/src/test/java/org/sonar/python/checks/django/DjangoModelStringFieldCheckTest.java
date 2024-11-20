/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

class DjangoModelStringFieldCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/django/djangoModelStringFieldCheck.py", new DjangoModelStringFieldCheck());
  }

  @Test
  void replaceQuickFixTest() {
    var check = new DjangoModelStringFieldCheck();
    var before = "from django.db import models\n" +
      "class NullFieldsModel(models.Model):\n" +
      "    name = models.CharField(max_length=50, null=True)";

    var after = "from django.db import models\n" +
      "class NullFieldsModel(models.Model):\n" +
      "    name = models.CharField(max_length=50, blank=True)";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"blank=True\"");
  }

  @Test
  void removeQuickFixTest() {
    var check = new DjangoModelStringFieldCheck();
    var before = "from django.db import models\n" +
      "class NullFieldsModel(models.Model):\n" +
      "    name = models.CharField(null=True, max_length=50, blank=True)";

    var after = "from django.db import models\n" +
      "class NullFieldsModel(models.Model):\n" +
      "    name = models.CharField(max_length=50, blank=True)";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Remove the \"null=true\" flag");

    before = "from django.db import models\n" +
      "class NullFieldsModel(models.Model):\n" +
      "    name = models.CharField(max_length=50, null=True, blank=True)";
    PythonQuickFixVerifier.verify(check, before, after);

    before = "from django.db import models\n" +
      "class NullFieldsModel(models.Model):\n" +
      "    name = models.CharField(max_length=50, blank=True, null=True)";
    PythonQuickFixVerifier.verify(check, before, after);

    before = "from django.db import models\n" +
      "class NullFieldsModel(models.Model):\n" +
      "    name = models.CharField(\n" +
      "        max_length=50,\n" +
      "        null=True,\n" +
      "        blank=True\n" +
      "    )";

    after = "from django.db import models\n" +
      "class NullFieldsModel(models.Model):\n" +
      "    name = models.CharField(\n" +
      "        max_length=50,\n" +
      "        blank=True\n" +
      "    )";
    PythonQuickFixVerifier.verify(check, before, after);
  }

}
