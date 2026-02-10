# chipstb
CHIPS-TB: Evaluating Tight-Binding Models For Metals, Semiconductors, and Insulators



<a name="install"></a>
# Installation

First create a conda environment:
Install miniforge https://github.com/conda-forge/miniforge

For example: 

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

Based on your system requirements, you'll get a file something like 'Miniforge3-XYZ'.

```
bash Miniforge3-$(uname)-$(uname -m).sh
```

Now, make a conda environment:

```
conda create --name my_chipstb python=3.11 -y
conda activate my_chipstb
```

```
git clone https://github.com/usnistgov/chipstb.git
cd chipstb
pip install -e .
```

Installing TB3Py

```
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz
tar -xvzf julia-1.11.5-linux-x86_64.tar.gz
export PATH=julia-1.11.5/bin:$PATH
julia --version
julia -e 'using Pkg; Pkg.add(["ThreeBodyTB", "Plots"]); Pkg.precompile()'
```

Installing DFTB+

```
conda install  mamba -y -q
mamba install 'dftbplus=*=nompi_*' -y
mamba install dftbplus-tools dftbplus-python -y
wget https://zenodo.org/records/14289468/files/ParameterSets.zip
unzip ParameterSets.zip
```

Installing SlaKoNet

```
git clone https://github.com/atomgptlab/slakonet.git
cd slakonet
pip install -e .
```

```
predict_slakonet.py  --jid JVASP-1002
```

<a name="run"></a>
# Run

```
python chipstb/run_chipstb.py --input_file chipstb/tb_input.json
```

Note: This project was originally developed under the github.com/usnistgov organization and is now maintained here by the lead developer.

# Reference

https://doi.org/10.1021/acs.jpcc.5c08042
