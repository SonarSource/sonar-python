from signxml import XMLVerifier

XMLVerifier().verify(xml)  # Noncompliant {{Change this code to only accept signatures computed from a trusted party.}}
XMLVerifier().verify(xml, x509_cert=cert)
XMLVerifier().verify(xml, cert_subject_name=cert_subject_name)
XMLVerifier().verify(xml, cert_resolver=cert_resolver)
XMLVerifier().verify(xml, ca_pem_file=ca_pem_file)
XMLVerifier().verify(xml, ca_path=ca_path)
XMLVerifier().verify(xml, hmac_key=hmac_key)

verify_kwargs = dict(require_x509=False, hmac_key=hmac_key, validate_schema=True, expect_config=sha1_ok)
XMLVerifier().verify(xml, **verify_kwargs)

verify_dict = {
    "hmac_key": hmac_key,
}
XMLVerifier().verify(xml, **verify_dict)

verify_indirect_dict = {
    "x509_cert": False
}
verify_indirect_dict_2 = {**verify_indirect_dict}
XMLVerifier().verify(xml, **verify_indirect_dict_2)


def fn():
    instanced_verifier = XMLVerifier()
    instanced_verifier.verify(xml)


# Coverage
"some_string".join("")
unrelated().verify(xml, **verify_kwargs)
unrelated.verify(xml, **verify_kwargs)
not_a_dict = some_func(a="b", c="d")
XMLVerifier().verify(xml, **not_a_dict)  # Noncompliant
some_str = "some_string"
XMLVerifier().verify(xml, **some_str)  # Noncompliant
dict_with_int_key = {1: "value"}
XMLVerifier().verify(xml, **dict_with_int_key)  # Noncompliant


def fp():
    from signxml import XMLVerifier
    my_dict = {}
    my_dict["x509_cert"] = False
    XMLVerifier().verify(xml, **my_dict)  # Noncompliant


# expect_config
from signxml import XMLVerifier, SignatureConfiguration, SignatureMethod

config = SignatureConfiguration(require_x509=False)
#                               ^^^^^^^^^^^^^^^^^^> {{Unsafe parameter set here}}
smth = XMLVerifier().verify(xml, hmac_key=hmac_key, expect_config=config)  # Noncompliant
#      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

config2 = SignatureConfiguration(
    require_x509=False,
    signature_methods=[SignatureMethod.HMAC_SHA256]
)

XMLVerifier().verify(xml, hmac_key=hmac_key, expect_config=config2)  # Noncompliant

config3 = SignatureConfiguration(signature_methods=[SignatureMethod.RSA_SHA224, SignatureMethod.HMAC_SHA256, SignatureMethod.SHA3_224_RSA_MGF1])
#                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^>                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>

smth2 = XMLVerifier().verify(xml, hmac_key=hmac_key, expect_config=config3)  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

XMLVerifier().verify(xml, hmac_key=hmac_key, expect_config=SignatureConfiguration(signature_methods=[SignatureMethod.RSA_SHA224]))  # Noncompliant

# Coverage
config4 = SignatureConfiguration(signature_methods=True)
XMLVerifier.verify(xml, hmac_key=hmac_key, expect_config=config4)

methods5 = [SignatureMethod.HMAC_SHA256, SignatureMethod.RSA_SHA256]
methods5 = []
config5 = SignatureConfiguration(signature_methods=methods5)
XMLVerifier().verify(xml, hmac_key=hmac_key, expect_config=config5)

config6 = SignatureConfiguration(signature_methods=3)
XMLVerifier().verify(xml, hmac_key=hmac_key, expect_config=config6)

XMLVerifier().verify(xml, hmac_key=hmac_key, expect_config=some_call())
XMLVerifier().verify(xml, hmac_key=hmac_key, expect_config=3)
