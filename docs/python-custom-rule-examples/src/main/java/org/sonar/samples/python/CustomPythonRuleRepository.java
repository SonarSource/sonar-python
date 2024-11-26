/*
 * Copyright (C) 2011-2024 SonarSource SA - mailto:info AT sonarsource DOT com
 * This code is released under [MIT No Attribution](https://opensource.org/licenses/MIT-0) license.
 */
package org.sonar.samples.python;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.server.rule.RulesDefinitionAnnotationLoader;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;
import org.sonar.samples.python.checks.CustomPythonSubscriptionCheck;
import org.sonar.samples.python.checks.CustomPythonVisitorCheck;

public class CustomPythonRuleRepository implements RulesDefinition, PythonCustomRuleRepository {

  @Override
  public void define(Context context) {
    NewRepository repository = context.createRepository(repositoryKey(), "py").setName("My custom repo");
    new RulesDefinitionAnnotationLoader().load(repository, checkClasses().toArray(new Class[] {}));
    Map<String, String> remediationCosts = new HashMap<>();
    remediationCosts.put(CustomPythonVisitorCheck.RULE_KEY, "5min");
    remediationCosts.put(CustomPythonSubscriptionCheck.RULE_KEY, "10min");
    repository.rules().forEach(rule -> rule.setDebtRemediationFunction(
      rule.debtRemediationFunctions().constantPerIssue(remediationCosts.get(rule.key()))));

    // Optionally override html description from annotation with content from html files
    repository.rules().forEach(rule -> rule.setHtmlDescription(loadResource("/org/sonar/l10n/python/rules/python/" + rule.key() + ".html")));
    repository.done();
  }

  @Override
  public String repositoryKey() {
    return "python-custom-rules";
  }

  @Override
  public List<Class> checkClasses() {
    return Arrays.asList(CustomPythonVisitorCheck.class, CustomPythonSubscriptionCheck.class);
  }

  private String loadResource(String path) {
    URL resource = getClass().getResource(path);
    if (resource == null) {
      throw new IllegalStateException("Resource not found: " + path);
    }
    ByteArrayOutputStream result = new ByteArrayOutputStream();
    try (InputStream in = resource.openStream()) {
      byte[] buffer = new byte[1024];
      for (int len = in.read(buffer); len != -1; len = in.read(buffer)) {
        result.write(buffer, 0, len);
      }
      return new String(result.toByteArray(), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException("Failed to read resource: " + path, e);
    }
  }
}
