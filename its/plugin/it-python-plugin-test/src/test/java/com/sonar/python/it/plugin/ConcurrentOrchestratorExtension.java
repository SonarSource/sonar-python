/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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
package com.sonar.python.it.plugin;

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.OrchestratorBuilder;
import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.build.SonarScannerInstaller;
import com.sonar.orchestrator.config.Configuration;
import com.sonar.orchestrator.container.SonarDistribution;
import com.sonar.orchestrator.server.StartupLogWatcher;
import com.sonar.orchestrator.util.System2;
import com.sonar.orchestrator.version.Version;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

public class ConcurrentOrchestratorExtension extends Orchestrator implements BeforeAllCallback {
  private static final AtomicInteger REQUESTED_ORCHESTRATORS_KEY = new AtomicInteger();
  private static final CountDownLatch IS_ORCHESTRATOR_READY = new CountDownLatch(1);

  ConcurrentOrchestratorExtension(Configuration config, SonarDistribution distribution, @Nullable StartupLogWatcher startupLogWatcher) {
    super(config, distribution, startupLogWatcher);
  }

  @Override
  public void beforeAll(ExtensionContext context) throws InterruptedException {
    if (REQUESTED_ORCHESTRATORS_KEY.getAndIncrement() == 0) {
      start();

      new SonarScannerInstaller(getConfiguration().locators()).install(Version.create(SonarScanner.DEFAULT_SCANNER_VERSION), getConfiguration().fileSystem().workspace());
      IS_ORCHESTRATOR_READY.countDown();

    } else {
      IS_ORCHESTRATOR_READY.await();
    }
  }


  public SonarScanner createSonarScanner() {
    return SonarScanner.create();
  }

  public static ConcurrentOrchestratorExtensionBuilder builderEnv() {
    return new ConcurrentOrchestratorExtensionBuilder(Configuration.createEnv());
  }

  public static class ConcurrentOrchestratorExtensionBuilder extends OrchestratorBuilder<ConcurrentOrchestratorExtensionBuilder, ConcurrentOrchestratorExtension> {
    ConcurrentOrchestratorExtensionBuilder(Configuration initialConfig) {
      this(initialConfig, System2.INSTANCE);
    }

    ConcurrentOrchestratorExtensionBuilder(Configuration initialConfig, System2 system2) {
      super(initialConfig, system2);
    }

    @Override
    protected ConcurrentOrchestratorExtension build(Configuration finalConfig, SonarDistribution distribution, StartupLogWatcher startupLogWatcher) {
      return new ConcurrentOrchestratorExtension(finalConfig, distribution, startupLogWatcher);
    }
  }
}
