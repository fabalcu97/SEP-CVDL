sudo apt -y install python3.11 python3.11-venv python3-pip
git clone --recurse-submodules git@github.com:werywjw/SEP-CVDL.git
cd SEP-CVDL
git checkout modifications
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd modified
jupyter lab --notebook-dir=.
