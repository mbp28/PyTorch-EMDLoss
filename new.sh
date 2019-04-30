rm -R dist/
rm -R build/
rm -R PyTorch_EMD.egg-info/
python setup.py install
python test_emd_loss.py
