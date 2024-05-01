package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.junit.jupiter.api.Assertions.*;

class SklearnPipelineParameterAreCorrectCheckTest {
  @Test
  void test(){
    PythonCheckVerifier.verify("src/test/resources/checks/sklearn_pipeline_parameter_are_correct.py", new SklearnPipelineParameterAreCorrectCheck());
  }

}