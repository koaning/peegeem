
# peegeem

> Probabilistic graphical models that are fun to work with <br>

This simple library is meant as a demo that shows what might be possible when you combine domain specific languages with bespoke UI tools. You might end up with a domain specific environment as a result!

![CleanShot 2025-05-02 at 16 23 22](https://github.com/user-attachments/assets/9618de8a-1f0b-49da-9055-0cbf124258ee)

## Install

You can install via pip.

```
uv pip install peegeem
```

## Usage 

This is how you might use this experiment. 

```python
from peegeem import DAG

# Define the DAG for the PGM, nodes is a list of column names, edges is a list of tuples
dag = DAG(nodes, edges, dataframe)

# Get variables out
outcome, smoker, age = dag.get_variables()

# Use variables to construct a probablistic query
P(outcome | (smoker == "Yes") & (age > 40))

# Latex utility, why not?
P.to_latex(outcome | (smoker == "Yes") & (age > 40))
```

The goal is to have an API that really closely mimics the math notation, so stuff like this:

$$ P(\\text{outcome} \\mid do(\\text{smoker}=\\texttt{Yes}), \\text{age}>40) $$

That means that we indeed also have a `do` function, though this needs more extensive testing. 

```python 
# You can also get crazy fancy 
P(A & B | C & do(D))
```

## Demo 

For a solid demo, download and run [this notebook](https://github.com/koaning/peegeem/blob/main/nbs/__init__.py) locally.

You can run the notebook in a sandboxed environment with uv. Navigate to the `nbs` directory and run:

```
uvx marimo edit --sandbox __init__.py
```
