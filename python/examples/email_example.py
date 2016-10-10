################################################################################
# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText

import getpass # To get the password from the command line, but not visible

smtpserver = 'smtp.gmail.com'

################################################################################
#### EDIT THESE ####
smtpuser = 'YOURGMAILADDRESS@gmail.com'  # for SMTP AUTH, set SMTP username here
smtppassword = "YOURPASSWORDHERE"

recipient = 'WHOAREYOUEMAILINGTHISTO@gmail.com'

# ALTERNATIVELY
#smtpuser = raw_input("What is the email address from which you will send this (must be GMail)? ")
#smtppasswd = getpass.getpass()
################################################################################

# Create a text/plain message

body = "This is the body text of a test email"

msg = MIMEText(body)
msg['Subject'] = "This is a test"
msg['From'] = smtpuser
msg['To'] = recipient

try:
    session = smtplib.SMTP('smtp.gmail.com',587)
    session.starttls()
    session.login(smtpuser,smtppasswd)
    session.sendmail(smtpuser, recipient, msg.as_string())
    print "Successfully sent email"
    session.quit()
except smtplib.SMTPException:
    print "Error: unable to send email"









