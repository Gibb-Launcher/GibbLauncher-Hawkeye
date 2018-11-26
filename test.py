import socket

s = socket.socket()
s.connect(('192.168.0.101' , 4444))
s.send("Você é um batatão! Errou quase Tudo.".encode())
s.close()