import smtplib
from email.mime.text import MIMEText



def send_email(content):

    _user = "*********"
    _pwd = "*********"
    _to = "*******"

    msg = MIMEText(content)
    msg["Subject"] = "Deep Learning Monitor"
    msg["From"]    = "Server"
    msg["To"]      = "iphone"

    try:
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)
        s.login(_user, _pwd)
        s.sendmail(_user, _to, msg.as_string())
        s.quit()
        print "Success!"
    except smtplib.SMTPException,e:
        print "Falied,%s"%e

if __name__ == '__main__':

    test = "test" + " = " + "34254867357" + "\n" + "hello email! \n"
    send_email(test)