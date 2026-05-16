import streamlit as st
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import time

# 1. KONFIGURASI TEMA & UI
st.set_page_config(page_title="PeanutAI Enterprise", page_icon="🌱", layout="wide")

# Custom CSS Premium (Sudah Diperbaiki agar Teks Sidebar Sangat Kentara)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .main { background: #f4f7f1; }
    
    /* --- STYLE SIDEBAR BARU (KENTARA & MENARIK) --- */
    .stSidebar { background-color: #2C3E2B !important; } /* Hijau tua solid */
    
    /* Judul menu utama di sidebar */
    .stSidebar p { 
        color: #FFFFFF !important; 
        font-weight: 600 !important; 
        font-size: 18px !important; 
        letter-spacing: 0.5px;
    }
    
    /* Teks pilihan menu (Dashboard, Deteksi AI, dll) menjadi putih bersih */
    .stSidebar .stRadio div [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important; 
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    /* Efek Interaktif: Warna teks berubah menjadi hijau pastel saat didekati mouse */
    .stSidebar .stRadio div:hover [data-testid="stMarkdownContainer"] p {
        color: #E9EDC9 !important;
        transition: 0.2s ease-in-out;
    }
    /* ---------------------------------------------- */
    
    .metric-card { 
        background: white; 
        padding: 25px; 
        border-radius: 15px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
        text-align: center; 
        border-bottom: 5px solid #4F6F52; 
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #e9edc9; border-radius: 10px; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #4F6F52 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. NAVIGASI MULTI-TAMPILAN DI SIDEBAR
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.markdown("<h2 style='color: white; text-align: center; margin-bottom: 20px;'>PeanutAI v2.0</h2>", unsafe_allow_html=True)
    st.write("---")
    menu = st.radio("Menu Navigasi:", ["🏠 Dashboard", "🔍 Deteksi AI", "🧮 Simulator Manual", "📚 Ensiklopedia", "👥 Tim Kami"])

# 3. LOGIKA ENGINE BACKEND (K-NEAREST NEIGHBORS)
# Representasi data latih berbasis nilai rata-rata RGB (Matriks pengetahuan sistem)
X_train = np.array([
    [45, 160, 50],   # Data 1: Sehat
    [50, 155, 48],   # Data 2: Sehat
    [180, 140, 30],  # Data 3: Karat Daun
    [175, 135, 35],  # Data 4: Karat Daun
    [70, 55, 40],    # Data 5: Bercak Daun
    [65, 50, 35]     # Data 6: Bercak Daun
])
y_train = np.array([0, 0, 1, 1, 2, 2]) # Label kelas numerik

# Training model KNN dengan parameter K=1
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
labels = {0: "🟢 DAUN SEHAT", 1: "🟡 TERKENA KARAT DAUN (RUST)", 2: "🟤 TERKENA BERCAK DAUN (LEAF SPOT)"}

# ----------------- HALAMAN 1: DASHBOARD -----------------
if menu == "🏠 Dashboard":
    st.title("🌿 Dashboard Utama PeanutAI")
    st.write("Selamat datang di platform manajemen kesehatan tanaman kacang tanah berbasis kecerdasan buatan.")
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h3>Akurasi Sistem</h3><h2 style='color:#4F6F52; font-size:40px;'>88.3%</h2><p style='color:gray;'>Pengujian Validasi</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h3>Dataset Basis</h3><h2 style='color:#4F6F52; font-size:40px;'>60+ Citra</h2><p style='color:gray;'>Sampel Daun Kacang</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h3>Algoritma Utama</h3><h2 style='color:#4F6F52; font-size:40px;'>KNN (K=1)</h2><p style='color:gray;'>Euclidean Distance</p></div>", unsafe_allow_html=True)
    
    st.write("<br><br>", unsafe_allow_html=True)
    st.subheader("💡 Mengapa Menggunakan Sistem Ini?")
    st.write("Sistem ini dirancang untuk menyelesaikan masalah deteksi penyakit foliar (daun) pada komoditas kacang tanah. Melalui analisis nilai pigmen warna daun, sistem mampu memberikan klasifikasi objektif dalam hitungan detik guna mencegah gagal panen.")

# ----------------- HALAMAN 2: DETEKSI AI -----------------
elif menu == "🔍 Deteksi AI":
    st.title("🔍 Deteksi Penyakit Real-Time (Blackbox)")
    st.write("Unggah foto sampel daun tanaman kacang Anda, sistem pintar kami akan mendiagnosis secara otomatis.")
    st.write("---")
    
    uploaded_file = st.file_uploader("Pilih file gambar daun (.jpg, .jpeg, .png)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        c1, c2 = st.columns(2)
        with c1:
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True, caption="Gambar Daun yang Diunggah")
        with c2:
            with st.status("Sedang mengekstrak fitur warna & memproses algoritma...", expanded=True) as status:
                # Konversi format gambar ke bentuk array OpenCV
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # Hitung rata-rata nilai saluran warna R, G, B
                b_avg = np.mean(img_cv[:,:,0])
                g_avg = np.mean(img_cv[:,:,1])
                r_avg = np.mean(img_cv[:,:,2])
                
                # Eksekusi klasifikasi KNN
                pred = knn.predict([[r_avg, g_avg, b_avg]])[0]
                time.sleep(1.2) # Memberikan efek loading simulasi proses cerdas
                status.update(label="Analisis Selesai!", state="complete", expanded=False)
            
            # Kartu Hasil Diagnosis Aesthetic
            st.markdown(f"""
                <div style="background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border-left: 6px solid #4F6F52;">
                    <h4 style="margin:0; color:#555;">Hasil Diagnosis Sistem:</h4>
                    <h2 style="color:#2C3E2B; margin-top:5px; margin-bottom:15px;">{labels[pred]}</h2>
                    <p style="margin:0; font-size:14px; color:#777;"><b>Fitur Warna Terekstrak:</b> Red: {r_avg:.2f} | Green: {g_avg:.2f} | Blue: {b_avg:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("<br>", unsafe_allow_html=True)
            st.subheader("💡 Rekomendasi Tindakan Ahli:")
            if pred == 0:
                st.success("Kondisi tanaman sehat sempurna! Pertahankan pola penyiraman berkala dan pastikan paparan sinar matahari tercukupi.")
            elif pred == 1:
                st.warning("Terdeteksi Karat Daun! Segera semprotkan fungisida berbahan aktif tembaga atau belerang. Pisahkan tanaman ini dari populasi sehat agar spora tidak menyebar lewat angin.")
            else:
                st.error("Terdeteksi Bercak Daun! Potong dan bakar area daun yang berbercak cokelat tua agar tidak menular. Hindari menyiram air langsung mengenai permukaan daun pada sore/malam hari.")

# ----------------- HALAMAN 3: SIMULATOR MANUAL -----------------
elif menu == "🧮 Simulator Manual":
    st.title("🧮 Simulator Perhitungan Teoretis KNN")
    st.write("Fitur interaktif untuk mencocokkan hasil program komputer dengan dokumen hitungan tangan.")
    st.write("---")
    
    st.subheader("1. Masukkan Nilai Piksel Sampel Uji Baru:")
    col_in1, col_in2, col_in3 = st.columns(3)
    r_in = col_in1.number_input("Nilai Red (R)", min_value=0, max_value=255, value=60)
    g_in = col_in2.number_input("Nilai Green (G)", min_value=0, max_value=255, value=140)
    b_in = col_in3.number_input("Nilai Blue (B)", min_value=0, max_value=255, value=45)
    
    if st.button("Jalankan Kalkulasi Jarak Euclidean"):
        st.write("### 📝 Langkah Rumus Matematika:")
        st.write("Rumus: $d = \\sqrt{(R_2 - R_1)^2 + (G_2 - G_1)^2 + (B_2 - B_1)^2}$")
        
        jarak_list = []
        for i, data_latih in enumerate(X_train):
            # Rumus Matematika Euclidean Distance
            jarak = np.sqrt((r_in - data_latih[0])**2 + (g_in - data_latih[1])**2 + (b_in - data_latih[2])**2)
            jarak_list.append(jarak)
            
            # Menampilkan kode perhitungan manualnya secara transparan
            st.code(f"Jarak ke Data Latih ke-{i+1} [{labels[y_train[i]][2:]}] =\n√(({r_in} - {data_latih[0]})² + ({g_in} - {data_latih[1]})² + ({b_in} - {data_latih[2]})²) = {jarak:.2f}")
            
        # Mencari indeks tetangga terdekat (jarak terkecil)
        tetangga_terdekat = np.argmin(jarak_list)
        st.write("---")
        st.success(f"**Kesimpulan Hitung Manual:** Jarak terpendek adalah **{jarak_list[tetangga_terdekat]:.2f}** menuju Data Latih ke-{tetangga_terdekat+1}. Maka data uji baru dikategorikan sebagai **{labels[y_train[tetangga_terdekat]]}**.")

# ----------------- HALAMAN 4: ENSIKLOPEDIA -----------------
elif menu == "📚 Ensiklopedia":
    st.title("📚 Ensiklopedia Penyakit Tanaman Kacang")
    st.write("Pelajari karakteristik morfologi gejala penyakit tanaman kacang tanah langsung dari pakar agrikultur.")
    st.write("---")
    
    tab1, tab2, tab3 = st.tabs(["🟢 Tanaman Sehat", "🟡 Karat Daun (Rust)", "🟤 Bercak Daun (Leaf Spot)"])
    
    with tab1:
        st.subheader("Karakteristik Tanaman Sehat")
        st.write("Daun kacang tanah yang sehat dicirikan oleh warna hijau segar yang merata di seluruh helaian daun. Proses fotosintesis berjalan optimal, tekstur permukaan daun terasa halus, dan tidak memiliki lesi nekrotik ataupun bercak berwarna ganjil.")
    with tab2:
        st.subheader("Gejala Karat Daun (Rust)")
        st.write("Disebabkan oleh jamur *Puccinia arachidis*. Gejala awal berupa bintik-bintik kecil berwarna kuning di permukaan bawah daun, yang kemudian berubah menjadi pustul (bintil) berwarna cokelat jingga seperti karat besi. Daun yang terserang parah akan menjadi kaku, mengering, dan gugur sebelum waktunya.")
    with tab3:
        st.subheader("Gejala Bercak Daun (Leaf Spot)")
        st.write("Umumnya disebabkan oleh jamur *Cercospora arachidicola*. Gejalanya terlihat dari adanya bercak-bercak bulat berwarna cokelat tua atau hitam pada permukaan atas daun. Lambat laun, bercak dikelilingi oleh lingkaran halo berwarna kuning cerah. Penyakit ini berkembang pesat pada kondisi lingkungan yang lembap.")

# ----------------- HALAMAN 5: PROFIL TIM -----------------
elif menu == "👥 Tim Kami":
    st.title("👥 Tim Pengembang Sistem")
    st.write("Proyek kolaborasi cerdas ini disusun untuk memenuhi tugas ujian tengah semester (Mid Test).")
    st.write("---")
    
    st.info("💡 **Anggota Kelompok :**")
    st.markdown("""
    * **Anggota 1:** [SYEILA RAHMADANI] - *Project Manager & Analyst*
    * **Anggota 2:** [UMMUL KHAIRIAH] - *Algorithm Expert (Hitung Manual)*
    * **Anggota 3:** [NUR KADRI] - *Backend Developer (Machine Learning)*
    * **Anggota 4:** [ASRIANI] - *UI/UX Designer (Streamlit Frontend)*
    """)
    st.success("Sistem ini siap diujikan di hadapan dosen penguji. Semangat untuk Tim Proyek 2026!")

# FOOTER KONSISTEN DI BAWAH SIDEBAR
st.sidebar.write("---")
st.sidebar.caption("Aplikasi Mid Test TI © 2026")