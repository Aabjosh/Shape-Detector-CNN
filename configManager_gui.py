import tkinter as tk
from tkinter import messagebox
import json
import os

FILE = "config.json"

# for getting the current configuration from the json file
def load_config():
    if os.path.exists(FILE):
        with open(FILE, "r") as f:
            return json.load(f)
    else:
        print("file not found.")
        return 0

# function for saving the new values by writing to the json
def save_config(config):
    with open(FILE, "w") as f:
        json.dump(config, f, indent=4)

# the class for the GUI
class interface:

    # constructor
    def __init__(self, root):
        self.root = root
        self.root.title("Set / Change Training Parameters")

        # use the "current" values by loading the full json and saving the values under ["current"] to a different self variable
        self.full_config = load_config()
        self.current_config = self.full_config["current"]

        # inputs, with local variables for the interface object (used to update the current values)
        tk.Label(root, text="Current Batch Size:").grid(row=0, column=0, sticky="e")
        self.v_batch = tk.StringVar(value=self.current_config["batch_size"])
        tk.Entry(root, textvariable=self.v_batch).grid(row=0, column=1)

        tk.Label(root, text="Current Epochs:").grid(row=1, column=0, sticky="e")
        self.v_epochs = tk.StringVar(value=self.current_config["epochs"])
        tk.Entry(root, textvariable=self.v_epochs).grid(row=1, column=1)

        tk.Label(root, text="Current Learning Rate:").grid(row=2, column=0, sticky="e")
        self.v_lr = tk.StringVar(value=self.current_config["learning_rate"])
        tk.Entry(root, textvariable=self.v_lr).grid(row=2, column=1)

        # buttons
        tk.Button(root, text="Use These Values", command=self.train_model).grid(row=3, column=0)
        tk.Button(root, text="Use Defaults", command=self.defaults).grid(row=3, column=1)

    # checks to see if inputs are valid, then updates self.full_config's ["current"] value to match the new ones
    def train_model(self):
        try:
            updated = {

                # use '.get()' since these are stringVars, not strings
                "batch_size": int(self.v_batch.get()),
                "epochs": int(self.v_epochs.get()),
                "learning_rate": float(self.v_lr.get())
            }    

            self.full_config["current"] = updated

            # save the new config to current, end the tkinter window
            save_config(self.full_config)
            self.root.destroy()

        # if not proper input, throw error
        except ValueError:
            messagebox.showerror("Invalid Input!", "Enter usable numbers.")

    # if defaults are requested, overwrite the ["current"] values with ["default"]
    def defaults(self):
        self.full_config["current"] = self.full_config["default"]
        save_config(self.full_config)
        self.root.destroy()

# function to use within 'model.py', in order to retrieve the new config, loops the window and creates the GUI interface. Once terminated, the ["current"] values are retrieved as dictionaries
def get_config():
    root = tk.Tk()
    app = interface(root)
    root.mainloop()