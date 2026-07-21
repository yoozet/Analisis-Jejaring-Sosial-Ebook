# Analisis Jejaring Sosial MOOC

## Identitas

- Nama: Fayyadh Fadhlurrohman
- Prodi: Teknologi Informasi
- Fakultas: Teknik
- Mata Kuliah: Analisis Jejaring Sosial
- Semester: 8B

## Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis jejaring sosial berbasis data MOOC menggunakan pendekatan analisis jaringan. Data yang digunakan diproses untuk menghasilkan visualisasi graf, pengukuran sentralitas, serta deteksi komunitas.

## Struktur Folder

- `analisis_mooc_fyd.py` — script utama untuk analisis jaringan
- `act-mooc/` — folder berisi dataset yang digunakan
- `visualisasi_mooc_jejaring_v2.png` — hasil visualisasi graf yang dihasilkan

## Persyaratan Sistem

- Python 3.9 atau versi lebih baru
- Sistem operasi Windows (panduan ini dibuat khusus untuk Windows)

## Panduan Instalasi

Berikut langkah instalasi yang dapat Anda lakukan di Windows:

### 1. Buka folder proyek

Buka folder proyek di Visual Studio Code atau terminal.

### 2. Buat virtual environment

Jalankan perintah berikut di terminal:

```bash
py -m venv .venv
```

### 3. Aktifkan virtual environment

Jika Anda menggunakan PowerShell, jalankan:

```bash
.\.venv\Scripts\Activate.ps1
```

Jika ada masalah terkait kebijakan PowerShell, jalankan:

```bash
Set-ExecutionPolicy -Scope Process RemoteSigned
```

### 4. Install dependency yang dibutuhkan

Install library Python berikut:

```bash
pip install pandas networkx matplotlib python-louvain
```

### 5. Jalankan program

Setelah instalasi selesai, jalankan script berikut:

```bash
py analisis_mooc_fyd.py
```

## Hasil yang Dihasilkan

Setelah program berjalan, Anda akan mendapatkan:

- output hasil analisis di terminal
- file visualisasi graf bernama `visualisasi_mooc_jejaring_v2.png`

## Catatan

Pastikan dataset berada di folder `act-mooc/` agar program dapat berjalan dengan benar.
