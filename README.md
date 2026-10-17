# Forex SMC Analyzer v4 — Web

Versi web sederhana yang memakai logika analisis dari `analyzer.py`. Tampilan difokuskan pada BUY/SELL/TUNGGU, Entry, SL, TP, RR, dan alasan singkat. Tombol analisis menjalankan proses hanya ketika ditekan. Tidak mengeksekusi order.

## Menjalankan lokal

1. Pasang Python 3.10 atau lebih baru.
2. Buka terminal di folder ini lalu jalankan:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variable `TWELVE_DATA_KEY` dengan API key Twelve Data. Jangan commit atau membagikan key.
4. Jalankan:
   ```bash
   streamlit run app.py
   ```

## Deploy gratis ke Streamlit Community Cloud

1. Buat akun GitHub dan repository baru, misalnya `smc-analyzer-web`.
2. Upload `app.py`, `analyzer.py`, `requirements.txt`, `.gitignore`, dan `README.md` ke repository.
3. Buka https://share.streamlit.io/ dan hubungkan GitHub.
4. Pilih repository, branch utama, dan file utama `app.py`, lalu deploy.
5. Di aplikasi Streamlit, buka **Settings → Secrets** dan tambahkan:
   ```toml
   TWELVE_DATA_KEY = "MASUKKAN_API_KEY_TWELVE_DATA_DI_SINI"
   ```
6. Simpan Secrets dan restart/reboot aplikasi jika diminta.

Jangan pernah menaruh API key di `app.py`, `analyzer.py`, repository publik, atau mengirimnya ke chat.

## Catatan penting

- Hosting gratis dapat tidur, memiliki batas penggunaan, atau berubah ketentuannya; tidak dijamin selalu tersedia 24/7.
- Paket API Twelve Data memiliki batas permintaan/kredit. Kegagalan mengambil data mungkin disebabkan limit, simbol yang tidak tersedia di paket, atau data belum cukup.
- Simbol `XAU/USD` bergantung pada dukungan paket/provider. Uji simbol di API Twelve Data sebelum mengandalkannya.
- Sinyal berasal dari aturan algoritmik pada Analyzer v4, bukan probabilitas kemenangan yang terjamin. Entry/SL/TP dan RR dihitung sebelum biaya trading sebagaimana dokumentasi kode asli.
