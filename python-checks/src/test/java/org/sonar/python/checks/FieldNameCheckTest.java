/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.PythonAstScanner;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.checks.CheckMessagesVerifier;

import java.io.File;

public class FieldNameCheckTest {

  @Test
  public void test() throws Exception {
    FieldNameCheck check = new FieldNameCheck();
    check.format = "^[_a-z][a-z0-9_]+$";
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/fieldName.py"), check);
    String message = "Rename this field \"%s\" to match the regular expression %s.";
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .next().atLine(3).withMessage(String.format(message, "myField", check.format))
        .next().atLine(10).withMessage(String.format(message, "myField1", check.format))
        .next().atLine(14).withMessage(String.format(message, "newField", check.format))
        .next().atLine(20).withMessage(String.format(message, "myField1", check.format))
        .next().atLine(20).withMessage(String.format(message, "myField2", check.format))
        .next().atLine(20).withMessage(String.format(message, "myField3", check.format))
        .next().atLine(21).withMessage(String.format(message, "newField", check.format))
        .next().atLine(24).withMessage(String.format(message, "Field1", check.format))
        .next().atLine(24).withMessage(String.format(message, "Field2", check.format))
        .next().atLine(27).withMessage(String.format(message, "Field3", check.format))
        .next().atLine(28).withMessage(String.format(message, "Field4", check.format))
        .next().atLine(28).withMessage(String.format(message, "Field5", check.format))
        .next().atLine(31).withMessage(String.format(message, "myField", check.format))
        .noMore();
  }

}
