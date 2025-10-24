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
package com.sonar.python.it;

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.OrchestratorBuilder;
import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.build.SonarScannerInstaller;
import com.sonar.orchestrator.config.Configuration;
import com.sonar.orchestrator.container.SonarDistribution;
import com.sonar.orchestrator.locator.Locators;
import com.sonar.orchestrator.server.StartupLogWatcher;
import com.sonar.orchestrator.util.System2;
import com.sonar.orchestrator.version.Version;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

public class ConcurrentOrchestratorExtension extends Orchestrator implements BeforeAllCallback {
  private static final Lock DOWNLOAD_LOCK = new ReentrantLock();

  private final AtomicInteger requestOrchestratorKey = new AtomicInteger();
  private final CountDownLatch isOrchestratorReady = new CountDownLatch(1);

  ConcurrentOrchestratorExtension(Configuration config, SonarDistribution distribution, @Nullable StartupLogWatcher startupLogWatcher) {
    super(config, distribution, startupLogWatcher);
  }

  @Override
  public void beforeAll(ExtensionContext context) throws InterruptedException {
    if (requestOrchestratorKey.getAndIncrement() == 0) {
      start();

      prepareOrchestrator();
    } else {
      waitUntilReady();
    }
  }

  private void prepareOrchestrator() {
    DOWNLOAD_LOCK.lock();
    try {
      installSonarScanner();
      isOrchestratorReady.countDown();
    } finally {
      DOWNLOAD_LOCK.unlock();
    }
  }

  private void installSonarScanner() {
    Locators locators = getConfiguration().locators();
    Version version = Version.create(SonarScanner.DEFAULT_SCANNER_VERSION);
    new SonarScannerInstaller(locators).install(version, getConfiguration().fileSystem().workspace());
  }


  public SonarScanner createSonarScanner() {
    return SonarScanner.create()
      .setProperty("sonar.scanner.skipJreProvisioning", "true");
  }

  public void waitUntilReady() throws InterruptedException {
    isOrchestratorReady.await();
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
