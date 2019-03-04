/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.checks.hotspots;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.python.semantic.Symbol;

@Rule(key = HashingDataCheck.CHECK_KEY)
public class HashingDataCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S4790";
  private static final String MESSAGE = "Make sure that hashing data is safe here.";
  private static final Set<String> questionableFunctions = immutableSet(
    "hashlib.new", "optparse.OptionParser", "cryptography.hazmat.primitives.hashes.Hash", "django.contrib.auth.hashers.make_password", "werkzeug.security.generate_password_hash");
  private static final Set<String> questionableHashlibAlgorithm = immutableSet(
    "blake2b", "blake2s", "md5", "pbkdf2_hmac", "sha1", "sha224",
    "sha256", "sha384", "sha3_224", "sha3_256", "sha3_384", "sha3_512",
    "sha512", "shake_128", "shake_256", "scrypt")
    .stream()
    .map(hasher -> "hashlib." + hasher)
    .collect(Collectors.toSet());

  private static final Set<String> questionablePasslibAlgorithm = immutableSet(
    "apr_md5_crypt", "argon2", "atlassian_pbkdf2_sha1", "bcrypt",
    "bcrypt_sha256", "bigcrypt", "bsd_nthash", "bsdi_crypt",
    "cisco_asa", "cisco_pix", "cisco_type7", "crypt16",
    "cta_pbkdf2_sha1", "des_crypt", "django_argon2", "django_bcrypt",
    "django_bcrypt_sha256", "django_des_crypt", "django_disabled",
    "django_pbkdf2_sha1", "django_pbkdf2_sha256", "django_salted_md5",
    "django_salted_sha1", "dlitz_pbkdf2_sha1", "fshp", "grub_pbkdf2_sha512",
    "hex_md4", "hex_md5", "hex_sha1", "hex_sha256", "hex_sha512",
    "htdigest", "ldap_bcrypt", "ldap_bsdi_crypt", "ldap_des_crypt", "ldap_hex_md5",
    "ldap_hex_sha1", "ldap_md5", "ldap_md5_crypt", "ldap_pbkdf2_sha1",
    "ldap_pbkdf2_sha256", "ldap_pbkdf2_sha512", "ldap_plaintext", "ldap_salted_md5",
    "ldap_salted_sha1", "ldap_sha1", "ldap_sha1_crypt", "ldap_sha256_crypt",
    "ldap_sha512_crypt", "lmhash", "md5_crypt", "msdcc", "msdcc2",
    "mssql2000", "mssql2005", "mysql323", "mysql41", "nthash", "oracle10",
    "oracle11", "pbkdf2_sha1", "pbkdf2_sha256", "pbkdf2_sha512", "phpass", "plaintext",
    "postgres_md5", "roundup_plaintext", "scram", "scrypt", "sha1_crypt", "sha256_crypt",
    "sha512_crypt", "sun_md5_crypt", "unix_disabled", "unix_fallback")
    .stream()
    .map(hasher -> "passlib.hash." + hasher)
    .collect(Collectors.toSet());


  private static final Set<String> questionableDjangoHashers = immutableSet(
    "PBKDF2PasswordHasher", "PBKDF2SHA1PasswordHasher", "Argon2PasswordHasher",
    "BCryptSHA256PasswordHasher", "BasePasswordHasher", "BCryptPasswordHasher", "SHA1PasswordHasher", "MD5PasswordHasher",
    "UnsaltedSHA1PasswordHasher", "UnsaltedMD5PasswordHasher", "CryptPasswordHasher")
    .stream()
    .map(hasher -> "django.contrib.auth.hashers." + hasher)
    .collect(Collectors.toSet());


  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(
      PythonGrammar.CALL_EXPR,
      PythonGrammar.ATTRIBUTE_REF,
      PythonGrammar.ATOM,
      PythonGrammar.EXPRESSION_STMT,
      PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    if (node.is(PythonGrammar.ATTRIBUTE_REF, PythonGrammar.ATOM)) {
      checkQuestionableHashingAlgorithm(node);

    } else if (node.is(PythonGrammar.EXPRESSION_STMT)) {
      checkOverwriteDjangoHashers(node);

    } else if (node.is(PythonGrammar.CLASSDEF)) {
      checkCreatingCustomHasher(node);

    } else if (node.is(PythonGrammar.CALL_EXPR)) {
      super.visitNode(node);
    }
  }

  @Override
  protected boolean isException(AstNode callExpression) {
    if (getQualifiedName(callExpression).equals("django.contrib.auth.hashers.make_password")) {
      AstNode argList = callExpression.getFirstChild(PythonGrammar.ARGLIST);
      if (argList != null) {
        return argList.getChildren(PythonGrammar.ARGUMENT).size() == 1;
      }
    }
    return false;
  }

  private void checkOverwriteDjangoHashers(AstNode expressionStatement) {
    AstNode lhsExpression = getLHSExpression(expressionStatement);
    if (lhsExpression == null) {
      return;
    }

    // checks for settings.PASSWORD_HASHERS = value
    AstNode attributeRef = lhsExpression.getFirstDescendant(PythonGrammar.ATTRIBUTE_REF);
    if (attributeRef != null) {
      AstNode atom = attributeRef.getFirstChild(PythonGrammar.ATOM);
      if (atom != null && getQualifiedName(atom).equals("django.conf.settings") &&
        attributeRef.getLastChild(PythonGrammar.NAME).getTokenValue().equals("PASSWORD_HASHERS")) {
        addIssue(expressionStatement, MESSAGE);
      }
    }

    // checks for PASSWORD_HASHERS = [] in a global_settings.py
    if (getContext().pythonFile().fileName().equals("global_settings.py") &&
      lhsExpression.getTokenValue().equals("PASSWORD_HASHERS")) {
      addIssue(expressionStatement, MESSAGE);
    }
  }

  private void checkQuestionableHashingAlgorithm(AstNode node) {
    String qualifiedName = getQualifiedName(node);
    if (qualifiedName.equals("cryptography.hazmat.primitives.hashes") && isHashesFunctionCall(node.getParent())) {
      addIssue(node.getParent(), MESSAGE);
    } else if (questionableHashlibAlgorithm.contains(qualifiedName) || questionablePasslibAlgorithm.contains(qualifiedName)) {
      addIssue(node, MESSAGE);
    }
  }

  private static boolean isHashesFunctionCall(@Nullable AstNode node) {
    if (node != null && node.is(PythonGrammar.ATTRIBUTE_REF)) {
      String propertyName = node.getLastChild(PythonGrammar.NAME).getTokenValue();
      AstNode parent = node.getParent();
      return propertyName.equals("Hash") && parent != null && parent.is(PythonGrammar.CALL_EXPR);
    }
    return false;
  }

  private static AstNode getLHSExpression(AstNode expressionStatement) {
    AstNode assign = expressionStatement.getFirstChild(PythonPunctuator.ASSIGN);
    if (assign != null) {
      return expressionStatement.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR);
    }
    return null;
  }


  private String getQualifiedName(AstNode node) {
    Symbol symbol = getContext().symbolTable().getSymbol(node);
    if (symbol != null) {
      return symbol.qualifiedName();
    }
    return "";
  }

  private void checkCreatingCustomHasher(AstNode classDef) {
    AstNode argList = classDef.getFirstChild(PythonGrammar.ARGLIST);
    if (argList != null) {
      argList.getDescendants(PythonGrammar.ATOM)
        .forEach(atom -> {
          if (questionableDjangoHashers.contains(getQualifiedName(atom))) {
            addIssue(atom, MESSAGE);
          }
        });
    }
  }

  @Override
  protected Set<String> functionsToCheck() {
    return questionableFunctions;
  }

  @Override
  protected String message() {
    return MESSAGE;
  }
}
