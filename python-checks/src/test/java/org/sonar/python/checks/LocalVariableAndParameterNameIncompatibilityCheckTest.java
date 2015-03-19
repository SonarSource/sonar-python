/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.PythonAstScanner;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.checks.CheckMessagesVerifier;

import java.io.File;

public class LocalVariableAndParameterNameIncompatibilityCheckTest {

  @Test
  public void test() throws Exception {
    LocalVariableAndParameterNameIncompatibilityCheck check = new LocalVariableAndParameterNameIncompatibilityCheck();
    check.format = "^[_a-z][a-z0-9_]+$";
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/localVariableAndParameterNameIncompatibility.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())

        .next().atLine(2).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "parameter", "inputPar2", check.format))
        .next().atLine(2).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "parameter", "inputPar3", check.format))
        .next().atLine(3).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "someName", check.format))

        .next().atLine(11).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "x", check.format))

        .next().atLine(16).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "parameter", "inputPar1", check.format))
        .next().atLine(16).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "parameter", "inputPar2", check.format))

        .next().atLine(20).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "x", check.format))
        .next().atLine(22).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "x", check.format))
        .next().atLine(22).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "y", check.format))

        .next().atLine(26).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "a", check.format))
        .next().atLine(26).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "b", check.format))
        .next().atLine(26).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "c", check.format))
        .next().atLine(27).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "d", check.format))
        .next().atLine(27).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "e", check.format))

        .next().atLine(30).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "parameter", "ID", check.format))
        .next().atLine(30).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "parameter", "ID2", check.format))

        .next().atLine(37).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "x", check.format))
        .next().atLine(45).withMessage(String.format(LocalVariableAndParameterNameIncompatibilityCheck.MESSAGE, "local variable", "counterName", check.format))
        .noMore();
  }

}
