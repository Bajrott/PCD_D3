# 🎓 Tugas Besar - Pengolahan Citra Digital

## 📌 Judul Proyek
**Ekstraksi Fitur Citra RGB untuk Klasifikasi Sampah (Plastik, Kertas, Organik)**

## 📖 Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan program pengolahan citra digital yang mampu mengekstraksi **fitur warna, bentuk, dan tekstur** dari citra RGB berisi objek sampah untuk tiga kategori utama: **Plastik**, **Kertas**, dan **Organik**.

## 🎯 Tujuan Proyek
- Membangun sistem ekstraksi fitur citra RGB dari dataset gambar.
- Mengimplementasikan tiga metode ekstraksi: warna (mean RGB), bentuk    (kontur), dan tekstur (GLCM).
- Mengklasifikasikan gambar berdasarkan fitur menggunakan pendekatan sederhana (perbandingan jarak).
- Menyediakan visualisasi hasil ekstraksi fitur.
- Mendokumentasikan hasil dan proses dalam laporan serta video demo.

## 🧠 Fitur yang Diekstraksi
1. Warna – Rata-rata warna dari citra RGB.
2. Bentuk – Kontur objek yang dideteksi dari citra grayscale.
3. Tekstur – Menggunakan metode GLCM untuk menghitung nilai kontras citra.

🗃️ Kategori Sampah
♻️ Anorganik
⚠️ B3 (Bahan Berbahaya & Beracun)
🍃 Organik

## 📦 Output Proyek
- ✅ Kode Program
- ✅ Dataset Citra
- ✅ Laporan Tertulis
- ✅ Video Demo
- ✅ Visualisasi hasil fitur setiap gambar

## 🗂 Struktur Proyek
📁 PCD_C3/
├── dataset_sampah/
│   ├── anorganik/
│   ├── b3/
│   └── organik/
├── ekstraksi_bentuk.py        
├── ekstraksi_teksture.py      
├── ekstraksi_warna.py        
├── PROJEK.PY                 
└── README.md     
📄 laporan.pdf
📄 demo_video.mp4

🔍 Cara Kerja
1. Inisialisasi Data Latih (Startup)
   Saat aplikasi dibuka, program mengekstrak fitur warna dan tekstur dari seluruh 62 gambar di `dataset_sampah`, lalu menyimpannya ke memori sebagai titik koordinat acuan.

2. Pre-processing & Segmentasi Citra
   Ketika gambar uji di-upload, sistem mengubah ukuran citra menjadi 250x250 piksel, mengonversinya ke grayscale, menerapkan thresholding biner, dan mendeteksi kontur objek untuk divisualisasikan langsung di GUI.

3. Ekstraksi Fitur Citra Uji
   Sistem mengekstrak nilai warna (Mean HSV & Standar Deviasi RGB) serta nilai tekstur (GLCM) dari gambar uji, kemudian menormalisasi seluruh nilainya ke rentang skala ketat 0.0 s.d 1.0.

4. Klasifikasi Weighted-KNN & Output
   Nilai fitur dihitung jarak Euclideannya terhadap data latih. Model melakukan voting tetangga terdekat ($K=5$) dengan pembobotan kuadrat jarak terbalik ($\frac{1}{d^2}$) untuk menampilkan hasil prediksi kategori sampah dan grafik persentasenya secara real-time di GUI.
   Project deteksi sampah menggunakan ekstraksi fitur citra.