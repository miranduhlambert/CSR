import tkinter as tk

def test_tkinter():
    root = tk.Tk()
    root.title("Tkinter Test")
    label = tk.Label(root, text="Tkinter is working!")
    label.pack()
    root.mainloop()

test_tkinter()