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
package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class SklearnPipelineSpecifyMemoryArgumentCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/sklearn_pipeline_specify_memory_argument.py", new SklearnPipelineSpecifyMemoryArgumentCheck());
  }

  @Test
  void test_quickfix_1(){
    PythonQuickFixVerifier.verify(
      new SklearnPipelineSpecifyMemoryArgumentCheck(),
      """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('classifier', LogisticRegression())
        ])
        """,
      """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('classifier', LogisticRegression())
        ], memory=None)
        """
    );
  }
  @Test
  void test_quickfix_2(){
    PythonQuickFixVerifier.verify(
      new SklearnPipelineSpecifyMemoryArgumentCheck(),
      """
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        pipeline = make_pipeline(StandardScaler(), LogisticRegression())
        """,
      """
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(), memory=None)
        """
    );
  }
}