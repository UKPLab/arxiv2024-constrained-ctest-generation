# Latin Hypercube Sampling

This script generates user study configurations with maximal distance and equal distribution.
To run this, first create a virtual environment (e.g., conda) and install the required packages:

    conda create --name=sampling python=3.10
    conda activate sampling
    pip install -r requirements.txt

The code can be run via:

    python fetch_minizinc_solutions.py

This generates some files in the `output` folder. First, a file containing all distances and possible combinations (`all_distances.dzn`) and second, one valid configuration (`final_combinations.json`). The code relies upon a constrainted optimization model found in the ```minizinc``` folder. For more information on [Minizinc](https://www.minizinc.org/); an open source constrained modeling language, please check their [documentation](https://www.minizinc.org/resources.html).

