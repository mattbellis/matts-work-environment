import numpy as np
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import base64
import ssl
import getpass
import os

################################################################################
# Reference
# https://julien.danjou.info/sending-emails-in-python-tutorial-code-examples/
# https://realpython.com/python-send-email/
################################################################################


########################################################################
sender_email = "matthew.bellis@gmail.com"
port = 465
password = getpass.getpass("Your password: ")

# FOR TESTING
receiver_email = "mbellis@siena.edu"

########################################################################

################################################################################
message = MIMEMultipart("alternative")
message["Subject"] = "Subject goes here"
message["From"] = sender_email
message["To"] = receiver_email

text = "Here is some text to send.\nAnd here is another line of text.\n\nSincerely,\nMe"

part1 = MIMEText(text, "plain")

message.attach(part1)

context = ssl.create_default_context()

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:

    server.login(sender_email, password)

    server.sendmail(sender_email, receiver_email, message.as_string())


