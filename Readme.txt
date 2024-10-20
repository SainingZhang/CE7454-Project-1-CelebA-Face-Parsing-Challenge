Files：
code is the code, test_result is the result on test images, best_model.pth is the best model, 
screen_shot.png is the mIoU of test reults, G2403905K_ZhangSaining_project1.pdf is the report.

Python: 3.12.7

Third-party libraries:
asttokens                   2.4.1
Brotli                      1.0.9
certifi                     2024.8.30
charset-normalizer          3.3.2
contourpy                   1.3.0
cycler                      0.12.1
decorator                   5.1.1
efficientnet_pytorch        0.7.1
executing                   2.1.0
filelock                    3.13.1
fonttools                   4.54.1
fsspec                      2024.9.0
huggingface-hub             0.25.1
idna                        3.7
ipdb                        0.13.13
ipython                     8.28.0
jedi                        0.19.1
Jinja2                      3.1.4
joblib                      1.4.2
kiwisolver                  1.4.7
MarkupSafe                  2.1.3
matplotlib                  3.9.2
matplotlib-inline           0.1.7
mkl_fft                     1.3.10
mkl_random                  1.2.7
mkl-service                 2.4.0
mpmath                      1.3.0
munch                       4.0.0
networkx                    3.3
numpy                       1.26.4
packaging                   24.1
parso                       0.8.4
pexpect                     4.9.0
pillow                      10.4.0
pip                         24.2
pretrainedmodels            0.7.4
prompt_toolkit              3.0.48
ptyprocess                  0.7.0
pure_eval                   0.2.3
Pygments                    2.18.0
pyparsing                   3.1.4
PySocks                     1.7.1
python-dateutil             2.9.0.post0
PyWavelets                  1.6.0
PyYAML                      6.0.1
requests                    2.32.3
safetensors                 0.4.5
scikit-learn                1.5.2
scipy                       1.14.1
segmentation-models-pytorch 0.3.4
setuptools                  75.1.0
six                         1.16.0
stack-data                  0.6.3
sympy                       1.13.2
threadpoolctl               3.5.0
timm                        0.9.7
torch                       2.4.1
torchaudio                  2.4.1
torchvision                 0.19.1
tqdm                        4.66.5
traitlets                   5.14.3
triton                      3.0.0
typing_extensions           4.11.0
urllib3                     2.2.3
wcwidth                     0.2.13
wheel                       0.44.0

Command:
data augmentation:
python data_augmentation.py

train:
python train.py

predict:
python predict.py

evaluate:
python evaluate.py