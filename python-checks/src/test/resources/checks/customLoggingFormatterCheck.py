import logging

def test_f_string():
    action = "purchase"
    logging.warning(f"{action}") # Noncompliant {{Use built-in logging formatting instead of using custom string formatting.}}
    #               ^^^^^^^^^^^

def test_string_format():
    logging.warning("$action".format("purchase")) # Noncompliant 

def test_log():
    action = "sell"
    logging.log(logging.INFO, f"{action}") # Noncompliant

def test_str_concat():
    action = "sell"
    logging.critical("action: " + action + " ") # Noncompliant

def test_direct_logging_usage():
    action, amount, item = "purchase", 42.50, "book"
    logging.info("User %s: %.2f %s", action, amount, item)  # Compliant - uses built-in formatting

# ------- COVERAGE -------

def test_other_loggers():
    def foo():
        pass

    logging.warn("$action".format("purchase")) # Noncompliant 
    logging.error("$action".format("purchase")) # Noncompliant 
    logging.exception("$action".format("purchase")) # Noncompliant
    logging.critical("$action".format("purchase")) # Noncompliant 
    logging.debug("$action".format("purchase")) # Noncompliant 
    logging.critical(f"action") # Compliant 
    logging.critical(f"") # Compliant 
    logging.info("") # Compliant 
    logging.info() # Compliant 
    logging.debug(foo()) # Compliant
    logging.critical(2 + 2) # Compliant
