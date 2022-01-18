## DINC-Net

#####   The trained model and denoising example for cardiopulmonary auscultation enhancement

  Paper : Cardiopulmonary Auscultation Enhancement with a Two-Stage Noise Cancellation Approach

### Requirement

- **Python 3.7**
- **PyTorch 1.7**
- **TorchAudio 0.7.1**

### Inference this model

```
python NoiseCancellationTest.py
```

### **Input:** 

​	 ./normal_NLMS/mix.wav

​	./normal_NLMS/noise.wav

### Output:

​	./result/restore.wav
