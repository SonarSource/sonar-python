def django_tests(subject, msg, from_email, to_email):
  from django.core.mail import send_mail, send_mass_mail

  send_mail(subject, msg, from_email, [to_email]) # Noncompliant {{Make sure that this email is sent in a safe manner.}}
# ^^^^^^^^^
  send_mass_mail((subject, msg, from_email, [to_email])) # Noncompliant

def smtplib_tests(from_email, to_email, msg):
    import smtplib
    server1 = smtplib.SMTP('localhost', 1025)
    server1.sendmail(from_email, to_email, msg) # Noncompliant
    server1.send_message(msg, from_email, to_email) # Noncompliant

    server2 = smtplib.SMTP_SSL('localhost', 1025)
    server2.sendmail(from_email, to_email, msg) # Noncompliant
    server2.send_message(msg, from_email, to_email) # Noncompliant

    class A:
        def sendmail(self): pass
        def send_message(self): pass

    a = A()
    a.sendmail(from_email, to_email, msg) # OK
    a.send_message(msg, from_email, to_email) # OK

def flask_mail_tests(app, msg, subject, to_email, body, from_email):
    from flask_mail import Mail, Connection
    mail = Mail(app)
    mail.send(msg) # Noncompliant
    mail.send_message(subject, [to_email], body, sender=from_email) # Noncompliant

    connection = Connection(mail)
    connection.send(msg) # Noncompliant
    connection.send_message(subject, [to_email], body, sender=from_email) # Noncompliant
