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
package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;


class SklearnCachedPipelineDontAccessTransformersCheckTest {

  @Test
  void test(){
    PythonCheckVerifier.verify("src/test/resources/checks/sklearn_cached_pipeline_dont_access_transformers.py", new SklearnCachedPipelineDontAccessTransformersCheck());
  }

  @Test
  void test_quickfix1(){
    PythonQuickFixVerifier.verify(
      new SklearnCachedPipelineDontAccessTransformersCheck(),
      """
        from sklearn.pipeline import Pipeline
        scaler = RobustScaler()
        knn = KNeighborsRegressor(n_neighbors=5)
            
        pipeline = Pipeline([
            ('scaler', scaler),
            ('knn', knn),
        ], memory="cache")
        print(scaler.center_)
        """,
      """
        from sklearn.pipeline import Pipeline
        scaler = RobustScaler()
        knn = KNeighborsRegressor(n_neighbors=5)
            
        pipeline = Pipeline([
            ('scaler', scaler),
            ('knn', knn),
        ], memory="cache")
        print(pipeline.named_steps["scaler"].center_)
        """
    );
    PythonQuickFixVerifier.verifyQuickFixMessages(
      new SklearnCachedPipelineDontAccessTransformersCheck(),
      """
        from sklearn.pipeline import Pipeline
        scaler = RobustScaler()
        knn = KNeighborsRegressor(n_neighbors=5)
            
        pipeline = Pipeline([
            ('scaler', scaler),
            ('knn', knn),
        ], memory="cache")
        print(scaler.center_)
        """,
      "Replace the direct access to the transformer with an access to the `named_steps` attribute of the pipeline."
    );
  }
}
