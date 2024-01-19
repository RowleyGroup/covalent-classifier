import marimo

__generated_with = "0.1.78"
app = marimo.App()


@app.cell
def __():
    import marimo as mo 
    import subprocess
    return mo, subprocess


@app.cell
def __(mo):
    input_string = mo.ui.text
    input_string
    return input_string,


@app.cell
def __(input_string, subprocess):
    command = f"python models/graph/gradcam.py {input_string}"
    subprocess.run(command, shell=True)
    return command,


if __name__ == "__main__":
    app.run()
