# Run these lines locally
scp ~/.ssh/cudo_instance_github cudo_compute:/root/.ssh/github
scp ~/.ssh/cudo_instance_github.pub cudo_compute:/root/.ssh/github.pub

# Run these lines on cudo_compute
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-distutils python3.11-venv -y
git clone --recurse-submodules git@github.com:fabalcu97/SEP-CVDL.git
cd SEP-CVDL
git checkout modifications
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd modified
jupyter lab --allow-root --ip 0.0.0.0 --notebook-dir=. & disown


# To copy things
# scp -r cudo_compute:/root/SEP-CVDL/modified/outputs/sc_wl_lr_ls_aw ./modified/outputs/sc_wl_lr_ls_aw
