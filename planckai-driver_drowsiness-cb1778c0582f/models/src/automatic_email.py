import smtplib
sender_email = "2210316501@gitam.in"
rec_email = "rohithajith123@gmail.com"
password = "rohith_official"
message = "Driver Drowzy !"
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(sender_email,password)
server.sendmail(sender_email, rec_email, message)