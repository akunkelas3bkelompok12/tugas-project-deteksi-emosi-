# Backend Assets

Folder ini berisi aset proses untuk model:

- `datasets/`: dataset training dan testing.
- `models/keras/`: model Keras sumber.
- `notebooks/`: notebook training.
- `legacy/`: salinan lama yang masih disimpan.
- `assets/`: aset pendukung.

Tidak ada server Flask di folder ini. Web Netlify berjalan dari folder `frontend/` dan melakukan prediksi di browser memakai TensorFlow.js.

