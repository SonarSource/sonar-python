/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
 * mailto:info AT sonarsource DOT com
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
package org.sonar.plugins.python.cpd;

import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.TokenType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.cpd.NewCpdTokens;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.TokenLocation;
import org.sonar.python.api.PythonTokenType;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.caching.CpdSerializer;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.caching.Caching.CPD_TOKENS_CACHE_KEY_PREFIX;

public class PythonCpdAnalyzer {

  private static final Logger LOG = Loggers.get(PythonCpdAnalyzer.class);

  private final SensorContext context;

  public PythonCpdAnalyzer(SensorContext context) {
    this.context = context;
  }

  public void pushCpdTokens(InputFile inputFile, PythonVisitorContext visitorContext) {
    Tree root = visitorContext.rootTree();
    if (root != null) {
      NewCpdTokens cpdTokens = context.newCpdTokens().onFile(inputFile);
      List<Token> tokens = TreeUtils.tokens(root);
      List<Token> tokensToCache = new ArrayList<>();
      for (int i = 0; i < tokens.size(); i++) {
        Token token = tokens.get(i);
        TokenType currentTokenType = token.type();
        TokenType nextTokenType = i + 1 < tokens.size() ? tokens.get(i + 1).type() : GenericTokenType.EOF;
        // INDENT/DEDENT could not be completely ignored during CPD see https://docs.python.org/3/reference/lexical_analysis.html#indentation
        // Just taking into account DEDENT is enough, but because the DEDENT token has an empty value, it's the
        // preceding new line which is added in its place to create a difference
        if (isNewLineWithIndentationChange(currentTokenType, nextTokenType) || !isIgnoredType(currentTokenType)) {
          TokenLocation location = new TokenLocation(token);
          cpdTokens.addToken(location.startLine(), location.startLineOffset(), location.endLine(), location.endLineOffset(), token.value());
          tokensToCache.add(token);
        }
      }
      saveTokensToCache(visitorContext, tokensToCache);
      cpdTokens.save();
    }
  }

  public boolean pushCachedCpdTokens(InputFile inputFile, CacheContext cacheContext) {
    String key = CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.key().replace('\\', '/');
    byte[] bytes = cacheContext.getReadCache().readBytes(key);
    if (bytes == null) {
      return false;
    }

    try {
      List<CpdSerializer.TokenInfo> tokens = CpdSerializer.fromBytes(bytes);

      NewCpdTokens cpdTokens = context.newCpdTokens().onFile(inputFile);
      tokens.forEach(tokenInfo ->
        cpdTokens.addToken(tokenInfo.startLine, tokenInfo.startLineOffset, tokenInfo.endLine, tokenInfo.endLineOffset, tokenInfo.value));
      cpdTokens.save();
      cacheContext.getWriteCache().copyFromPrevious(key);
      return true;
    } catch (IOException | ClassNotFoundException | ClassCastException e) {
      LOG.warn("Failed to deserialize CPD tokens ({}: {})", e.getClass().getSimpleName(), e.getMessage());
    }

    return false;
  }

  private static void saveTokensToCache(PythonVisitorContext visitorContext, List<Token> tokensToCache) {
    CacheContext cacheContext = visitorContext.cacheContext();
    if (!cacheContext.isCacheEnabled()) {
      return;
    }

    try {
      byte[] bytes = CpdSerializer.toBytes(tokensToCache);
      cacheContext.getWriteCache().write(CPD_TOKENS_CACHE_KEY_PREFIX + visitorContext.pythonFile().key().replace('\\', '/'), bytes);
    } catch (Exception e) {
      LOG.warn("Could not write CPD tokens to cache ({}: {})", e.getClass().getSimpleName(), e.getMessage());
    }
  }

  private static boolean isNewLineWithIndentationChange(TokenType currentTokenType, TokenType nextTokenType) {
    return currentTokenType.equals(PythonTokenType.NEWLINE) && nextTokenType.equals(PythonTokenType.DEDENT);
  }

  private static boolean isIgnoredType(TokenType type) {
    return type.equals(PythonTokenType.NEWLINE) ||
      type.equals(PythonTokenType.DEDENT) ||
      type.equals(PythonTokenType.INDENT) ||
      type.equals(GenericTokenType.EOF);
  }

}
