// Membuka file explorer pas tombol "Pilih Gambar" diklik
document.getElementById('btnChoose').addEventListener('click', function() {
    document.getElementById('fileInput').click();
});

// Aksi saat gambar sukses dipilih
document.getElementById('fileInput').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        // Nyalakan tombol analisis sekarang
        document.getElementById('btnAnalyze').removeAttribute('disabled');
        
        // Buat preview gambar
        let reader = new FileReader();
        reader.onload = function(event) {
            let preview = document.getElementById('previewImg');
            preview.src = event.target.result;
            preview.classList.remove('hidden');
            document.getElementById('dropZoneInner').classList.add('hidden');
        }
        reader.readAsDataURL(e.target.files[0]);
    }
});

// Tembak Ajax ke Flask pas tombol analisis diklik
document.getElementById('btnAnalyze').addEventListener('click', function() {
    let fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) return;

    let formData = new FormData();
    formData.append('file', fileInput.files[0]);

    let progressLabel = document.getElementById('progressLabel');
    let progressBar = document.getElementById('progressBar');
    let progressPct = document.getElementById('progressPct');
    
    progressLabel.innerText = "Mengekstrak multi-fitur citra (Warna, Bentuk, Tekstur)...";
    progressBar.style.width = "60%";
    progressPct.innerText = "60%";

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // Selesaikan Progress bar
        progressBar.style.width = "100%";
        progressPct.innerText = "100%";
        progressLabel.innerText = "Analisis Fitur Selesai!";

        // Buka Result Card
        let resultCard = document.getElementById('resultCard');
        if (resultCard) resultCard.classList.remove('hidden');
        
        document.getElementById('resultCategory').innerText = data.prediksi;
        document.getElementById('resultConfidence').innerText = "Klasifikasi Jarak Terdekat KNN Berhasil";

        // Update progress bar nilai kelas sampah
        document.getElementById('barOrganik').style.width = data.skor_visual.organik + "%";
        document.getElementById('pctOrganik').innerText = data.skor_visual.organik + "%";

        document.getElementById('barNon').style.width = data.skor_visual.anorganik + "%";
        document.getElementById('pctNon').innerText = data.skor_visual.anorganik + "%";

        document.getElementById('barB3').style.width = data.skor_visual.d3 + "%";
        document.getElementById('pctB3').innerText = data.skor_visual.d3 + "%";

        // Buka blok visualisasi gambar OpenCV kolom vertikal
        let vizBox = document.getElementById('featuresViz');
        if (vizBox) vizBox.style.display = 'block';
        
        // Injeksi file image hasil pemrosesan
        let imgGray = document.getElementById('canvasGray');
        let imgThresh = document.getElementById('canvasEdge');
        let imgKontur = document.getElementById('canvasHist');

        if(imgGray) imgGray.src = data.images.gray;
        if(imgThresh) imgThresh.src = data.images.thresh;
        if(imgKontur) imgKontur.src = data.images.kontur;
    })
    .catch(error => {
        console.error('Error:', error);
        progressLabel.innerText = "Terjadi kegagalan sistem web.";
    });
});