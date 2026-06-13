# Deteksi Emosional Wajah

Project ini sudah dirapikan menjadi Netlify static-only. Tidak ada backend Flask/server yang dipakai untuk menjalankan web.

## Struktur Folder

- `frontend/`: aplikasi web yang di-deploy ke Netlify.
- `frontend/models/`: model TensorFlow.js untuk prediksi langsung di browser.
- `backend/`: dataset, notebook training, model Keras sumber, dan aset proses. Folder ini bukan server Flask.
- `.tools/`: Node.js, npm, dan Netlify CLI lokal.

## Jalankan Lokal

```powershell
py -3.10 -m http.server 8080 -d frontend
```

Buka:

```text
http://127.0.0.1:8080
```

## Deploy Netlify

Login satu kali:

```powershell
.\run-netlify.cmd login
```

Deploy production:

```powershell
.\run-netlify.cmd deploy --prod --dir=frontend
```

Atau:

```powershell
.\.tools\node\npm.cmd run netlify:deploy
```

Di dashboard Netlify:

- Build command: kosongkan
- Publish directory: `frontend`

## Catatan

Dataset dan notebook training tetap disimpan di `backend/`, tetapi web Netlify tidak memuat dataset langsung saat prediksi. Prediksi memakai model TensorFlow.js yang sudah ada di `frontend/models/`.
