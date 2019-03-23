from sip_lstm import generate_sip
from tkinter import *
import datetime
import os
import tkinter.font

def change_sip():
	sip_text.set('loading...')
	window.update()
	sip_text.set(generate_sip(['']))

def reset_sip():
	sip_text.set('')
	window.update()
	sip_text.set(default_sip)

def save_sip():
	if not os.path.exists(saved_sips_dir):
		os.makedirs(saved_sips_dir)
	file = open('%s/sip_%d.txt' % (saved_sips_dir, (datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds()), 'w')
	file.write(sip_text.get())
	file.close()

default_sip = 'authored by computer...\nshort Issa poems for you\nnow, click the button'
saved_sips_dir = 'saved_sips'

window = Tk()
window.title('SIP Generator')
window.iconbitmap('icons/issa2.ico')
window.configure(background='black')
window_size = 1000
window.geometry('%dx%d' % (window_size, window_size))

sip_text = StringVar()
sip_text.set(default_sip)
sip_font = tkinter.font.Font(family='Helvetica', size=36, weight='bold')
sip_label = Label(window, textvariable=sip_text, font=sip_font, bg='black', fg='white')
sip_label.place(relx=0.5, rely=0.4, anchor='c')

generate_font = tkinter.font.Font(family='Helvetica', size=24, weight='bold')
generate_button = Button(window, text='Take a SIP', font=generate_font, command=change_sip)
generate_button.place(relx=0.5, rely=0.8, anchor='c')

small_font = tkinter.font.Font(family='Helvetica', size=12, weight='bold')
sub_label = Label(window, text='(Short Issa Poem)', font=small_font, bg='black', fg='white')
sub_label.place(relx=0.5, rely=0.85, anchor='c')

reset_button = Button(window, text='Reset SIP', font=small_font, command=reset_sip)
reset_button.place(relx=0.15, rely=0.9, anchor='c')

save_button = Button(window, text='Save SIP', font=small_font, command=save_sip)
save_button.place(relx=0.85, rely=0.9, anchor='c')

mainloop()