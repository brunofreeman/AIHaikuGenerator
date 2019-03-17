from tkinter import *
import tkinter.font

window = Tk()
window.title("Issa Generator")
window.configure(background="black")
window_size = 750
window.geometry(str(window_size) + "x" + str(window_size))
window.resizable(width=False, height=False)
'''
rc = 0
num_rc = 5
while rc <= num_rc:
	window.rowconfigure(rc, minsize=int(window_size / num_rc))
	window.columnconfigure(rc, minsize=int(window_size / num_rc))
	rc += 1
'''
haiku_text = StringVar()
haiku_text.set("haiku-- first line\nhaiku-- second line\nhaiku-- third line")
haiku_font = tkinter.font.Font(family="Helvetica", size=36, weight="bold")
haiku_label = Label(window, textvariable=haiku_text, font=haiku_font, bg="black", fg="white")
haiku_label.place(relx=0.5, rely=0.4, anchor="c")

def generate_new_haiku():
	haiku_text.set("new haiku")

button_font = tkinter.font.Font(family="Helvetica", size=24, weight="bold")
generate_button = Button(window, text="Generate a SIP", font=button_font, command=generate_new_haiku)
generate_button.place(relx=0.5, rely=0.9, anchor="c")

sub_font = tkinter.font.Font(family="Helvetica", size=12,)
sub_label = Label(window, text="(Short Issa Poem)", font=sub_font, bg="black", fg="white")
sub_label.place(relx=0.5, rely=0.95, anchor="c")

mainloop()