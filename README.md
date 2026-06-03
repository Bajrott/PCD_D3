# 🎓 Tugas Besar - Pengolahan Citra Digital (Semester 4)
> **Program Studi Informatika — Fakultas Teknologi Industri** > **Institut Teknologi Nasional (Itenas), Bandung — 2026**

---

## 📌 Judul Proyek
**Sistem Klasifikasi Jenis Sampah Berbasis Aplikasi Web Menggunakan Ekstraksi Multi-Fitur dan Algoritma Weighted K-Nearest Neighbor (W-KNN)**

---

## 📖 Deskripsi Proyek
Proyek ini dikembangkan untuk mengatasi permasalahan pemilahan sampah rumah tangga secara otomatis menggunakan pendekatan **Pengolahan Citra Digital (PCD)** dan *Machine Learning* sederhana. Sistem ini mampu mengidentifikasi dan mengklasifikasikan objek sampah ke dalam tiga kategori utama: **Organik**, **Anorganik**, dan **B3 (Bahan Berbahaya dan Beracun)**.

Berbeda dengan aplikasi desktop tradisional, sistem ini mengintegrasikan core pemrosesan citra **OpenCV (Python)** dengan interface modern berbasis **Web Application (Flask, HTML5, CSS3 Glassmorphism, dan JavaScript)**. Pemilihan kategori didasarkan pada kombinasi ekstraksi fitur visual yang komprehensif, meliputi analisis karakteristik warna, geometri bentuk, dan pola tekstur permukaan objek.

---

## 👥 Anggota Kelompok (Kelas C3)
* **152023021** – Azzra Rizky Pratama
* **152024001** – Mahabayubadra Ardayanta
* **152024026** – Gregorius David Stu
* **152024040** – Nur Hayyu Fadillah
* **152024167** – Muhamad Ramdan Fauzi

---

## 🎯 Tujuan Proyek
* Membangun pipeline segmentasi citra digital yang adaptif untuk memisahkan objek sampah dari latar belakang.
* Mengimplementasikan teknik ekstraksi multi-fitur: **Warna** (Mean RGB/HSV & Dominant Color K-Means), **Bentuk** (Otsu Thresholding, Kontur, Hu Moments), dan **Tekstur** (GLCM & LBP).
* Menerapkan algoritma **Weighted K-Nearest Neighbor (W-KNN)** dengan pembobotan kuadrat jarak terbalik murni untuk klasifikasi objek citra uji secara *real-time*.
* Menyediakan visualisasi tahapan pemrosesan citra (*Grayscale*, *Thresholding*, *Contour Tracking*) secara interaktif pada antarmuka web yang responsif.

---

## 🧠 Fitur & Ekstraksi Citra Ilmiah
Sistem mengekstrak nilai-nilai fisis citra yang kemudian dinormalisasi ke dalam vektor fitur:
1.  **Fitur Warna:** Menghitung rata-rata (*mean*) dan standar deviasi pada ruang warna RGB dan HSV, ekstraksi warna dominan via *K-Means Clustering*, serta analisis histogram distribusi warna untuk mendongkrak akurasi label sensitif (seperti B3).
2.  **Fitur Bentuk:** Melakukan ekstraksi geometri objek meliputi *Area*, *Perimeter*, *Eccentricity*, *Solidity*, *Extent*, Rasio Aspek, dan 5 komponen *Hu Moments* dari region objek terbesar.
3.  **Fitur Tekstur:** Memanfaatkan metode *Gray-Level Co-occurrence Matrix* (GLCM) untuk menghitung parameter *Contrast*, *Dissimilarity*, *Homogeneity*, *Energy*, serta ekstraksi pola tekstur lokal menggunakan *Local Binary Pattern* (LBP).

---

## 🗃️ Kategori Klasifikasi Sampah
* 🍃 **Organik:** Sampah alamiah yang mudah terurai (Sisa makanan, dedaunan, ranting, ampas kopi).
* ♻️ **Anorganik:** Sampah non-biologis yang dapat didaur ulang (Plastik, logam, kardus, kaca/botol).
* ⚠️ **B3 (Bahan Berbahaya & Beracun):** Limbah yang memerlukan penanganan khusus (Baterai, obat-obatan, limbah elektronik, kaleng cat/kimia).

---

## 📦 Output Proyek
* `✅` **Kode Program:** Backend berbasis Python Flask (`PROJEK.PY`) dan Frontend responsive (`templates/index.html`, `static/app.js`, `static/style.css`).
* `✅` **Dataset Citra:** Terdiri dari total gambar latih yang terbagi ke dalam folder kategori `organik`, `anorganik`, dan `b3`.
* `✅` **Laporan Tertulis:** Dokumentasi ilmiah format PDF/Docx lengkap dari bab pendahuluan hingga pengujian.
* `✅` **Visualisasi Interaktif:** Tampilan *step-by-step* hasil filter OpenCV langsung pada halaman web aplikasi.

---

## 🗂️ Struktur Direktori Proyek
```text
📁 PCD_C3-MAIN/
│
├── 📁 dataset_sampah/          # Folder Dataset Latih
│   ├── 📁 anorganik/           # Sampah Plastik, Kaca, Logam, Kardus
│   ├── 📁 b3/                  # Baterai, Kaleng Kimia, Elektronik
│   └── 📁 organik/             # Sisa Makanan, Daun, Organik Alamiah
│
├── 📁 templates/               # Engine Template HTML Flask
│   └── 📄 index.html          # Antarmuka Dashboard Web Aplikasi
│
├── 📁 static/                  # File Aset Statis Web
│   ├── 📄 style.css            # Desain UI Mode Gelap & Glassmorphism
│   ├── 📄 app.js               # Logic AJAX Fetch, Progress Bar & Renderer Citra
│   └── 📁 uploads/             # Tempat Penyimpanan Citra Hasil Olahan OpenCV
│
├── 📄 ekstraksi_bentuk.py      # Ekstraksi Area, Perimeter, Solidity, Hu Moments
├── 📄 ekstraksi_teksture.py    # Ekstraksi GLCM (Contrast, Homogeneity) & LBP
├── 📄 ekstraksi_warna.py       # Ekstraksi Mean RGB/HSV & K-Means Color Clustering
├── 📄 PROJEK.PY                # File Utama Server Flask & Engine KNN Classifier
└── 📄 README.md                # Dokumentasi Proyek
