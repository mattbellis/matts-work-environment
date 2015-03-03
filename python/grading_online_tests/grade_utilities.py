# Import smtplib for the actual sending function
import smtplib
# Import the email modules we'll need
from email.mime.text import MIMEText

################################################################################
def email_grade_summaries(email_address,msg_from,msg_subject,msg_body,password="xxx"):

    ################################################################################
    # Use my GMail account
    ################################################################################
    smtpserver = 'smtp.gmail.com'
    #smtpuser = 'matthew.bellis@gmail.com'  # for SMTP AUTH, set SMTP username here
    smtpuser = msg_from  # for SMTP AUTH, set SMTP username here
    smtppasswd = password  # for SMTP AUTH, set SMTP password here
    #me = 'matthew.bellis@gmail.com'
    me = msg_from

    # Create a text/plain message
    msg = MIMEText(msg_body)
    msg['Subject'] = '%s' % (msg_subject)
    msg['From'] = me
    msg['To'] = email_address

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    try:
        session = smtplib.SMTP('smtp.gmail.com',587)
        session.starttls()
        session.login(smtpuser,smtppasswd)
        session.sendmail(me, email_address, msg.as_string())
        print "Successfully sent email"
        session.quit()
    except smtplib.SMTPException:
        print "Error: unable to send email"

