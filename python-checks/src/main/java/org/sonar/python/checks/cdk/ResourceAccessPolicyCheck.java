/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.cdk;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionCheck;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;

import static org.sonar.python.checks.cdk.CdkPredicate.isWildcard;

@Rule(key = "S6304")
public class ResourceAccessPolicyCheck extends AbstractIamPolicyStatementCheck {

  private static final Logger LOG = LoggerFactory.getLogger(ResourceAccessPolicyCheck.class);
  private static final String MESSAGE = "Make sure granting access to all resources is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect";
  private static final Map<String, Set<String>> CACHED_RESOURCES = new ConcurrentHashMap<>();
  // visible for testing
  String resourceNameSensitiveAwsActions = "ResourceAccessPolicyCheck.txt";
  private Set<String> sensitiveAwsActions = null;

  void init() {
    sensitiveAwsActions = CACHED_RESOURCES.computeIfAbsent(resourceNameSensitiveAwsActions, ResourceAccessPolicyCheck::loadResourceWrapper);

  }
  @Override
  public void initialize(SubscriptionCheck.Context context) {
    super.initialize(context);
    init();
  }

  private static Set<String> loadResourceWrapper(String resourceName) {
    try {
      return loadResource(resourceName);
    } catch (IOException e) {
      LOG.error("Couldn't load resource '{}', rule [S6304] ResourceAccessPolicyCheck will be disabled.", resourceName, e);
      return Set.of();
    }
  }

  private static Set<String> loadResource(String resourceName) throws IOException {
    try (InputStream is = ResourceAccessPolicyCheck.class.getResourceAsStream(resourceName)) {
      if (is == null) {
        throw new IOException("Cannot find resource file '" + resourceName + "'");
      }
      try (InputStreamReader isr = new InputStreamReader(is, StandardCharsets.UTF_8);
           BufferedReader br = new BufferedReader(isr)) {
        List<String> result = new ArrayList<>();
        String line;
        while((line = br.readLine()) != null) {
          result.add(line);
        }
        return new HashSet<>(result);
      }
    }
  }

  @Override
  protected void checkAllowingPolicyStatement(PolicyStatement policyStatement) {
    CdkUtils.ExpressionFlow actions = policyStatement.actions();
    CdkUtils.ExpressionFlow resources = policyStatement.resources();

    if (resources == null || actions == null || !isSensitiveAction(actions)) {
      return;
    }

    Optional.ofNullable(getSensitiveExpression(resources, isWildcard()))
      .ifPresent(wildcard -> reportWildcardResourceAndEffect(wildcard, policyStatement.effect()));
  }

  private boolean isSensitiveAction(ExpressionFlow actions) {
    return getSensitiveExpression(actions, inSensitiveSet()) != null;
  }

  public Predicate<Expression> inSensitiveSet() {
    return expression -> CdkUtils.getString(expression)
      .filter(sensitiveAwsActions::contains).isPresent();
  }

  private static void reportWildcardResourceAndEffect(ExpressionFlow wildcard, @Nullable ExpressionFlow effect) {
    PreciseIssue issue = wildcard.ctx().addIssue(wildcard.getLast(), MESSAGE);
    if (effect != null) {
      issue.secondary(effect.asSecondaryLocation(SECONDARY_MESSAGE));
    }
  }
}
