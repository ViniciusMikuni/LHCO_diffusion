# Diffusion-based Model for Resonand Anomaly Detection

To train the model you can run:
```bash
python train.py
```

To sample from the trained model you can run:
```bash
python plot_jet.py  --sample --SR
```
The ```--SR``` flags determines if the background sample is done in the signal region (with --SR flag) or in the side-band region (without the --SE flag).

Drop the ```--sample``` to reproduce the plots without having to regenerate new files.

A classifier used to separate generated samples from data can be trained using:

```bash
python classify.py --SR
```

Where the ```--SR``` flag again determines where the backgrounds are sampled from.