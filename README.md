# Laporan Proyek Machine Learning - M.Wahyu Elfander

## Domain Proyek

Industri mobil bekas telah menjadi segmen yang signifikan dalam pasar otomotif global. Dalam beberapa tahun terakhir, permintaan untuk mobil bekas telah meningkat karena berbagai alasan, termasuk kenaikan harga mobil baru, pilihan yang lebih luas untuk konsumen, serta ekonomi yang tidak stabil yang mendorong konsumen untuk mencari opsi yang lebih terjangkau.

Predictive analytics adalah teknik yang digunakan untuk membuat prediksi tentang hasil masa depan berdasarkan data historis. Dalam konteks pasar mobil bekas, predictive analytics dapat membantu menentukan harga jual yang optimal berdasarkan berbagai variabel yang mempengaruhi nilai sebuah mobil. Beberapa faktor utama yang sering dipertimbangkan dalam menentukan harga mobil bekas meliputi tahun pembeleian,kilometer tempuh,owner,dll.

Tujuan utama dari analisis ini adalah untuk mengembangkan model prediktif yang dapat memperkirakan harga jual mobil bekas secara akurat berdasarkan data historis yang tersedia. Dengan menggunakan teknik KNN, Random Forrest, dan Boosting Algorithm, kita dapat mengidentifikasi pola dan hubungan antara variabel independen (faktor-faktor yang mempengaruhi harga) dan variabel dependen (harga mobil).

Selain menerapkan predictive analytics untuk prediksi, tujuan dari analisis ini adalah untuk melakukan visualisasi data yang mempengaruhi harga mobil bekas pada dataset yang akan di analisis nanti. Visualisasi yang akan di tampilkan adalah korelasi antara harga dan tahun, korelasi antara harga dan fitur owner. Karena fitur owner dan tahun memiliki korelasi yang besar terhadap harga ketika analisis dilakukan.

Dengan menggunakan predictive analytics, perusahaan dapat mengoptimalkan strategi penjualan mereka, memberikan harga yang kompetitif, dan meningkatkan kepuasan pelanggan dengan memberikan estimasi harga yang realistis dan transparan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
1. Untuk melakukan prediksi by data, maka harus mencari dataset yang relefan dan lengkap
2. Untuk mempermudah perusahaan, sistem yang dibangun harus bisa memprediksi harga mobil yang beredar dipasaran berdasarkan fitur-fitur yang mempengaruhi harga
3. Sistem prediksi harga mobil yang akan dibangun menggunakan pendekatan predictive analytics 

- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.

  - Referensi 1: [Car ownership motivations among undergraduate students in China, Indonesia, Japan, Lebanon, Netherlands, Taiwan, and USA](https://doi.org/10.1007/s11116-014-9548-z) 
  - Refrensi 2 : [Used Car Price Prediction using K-Nearest Neighbor Based
Model](https://doi.org/10.29027/ijirase.v4.i2.2020.629-632)

## Business Understanding

### Problem Statements

- Mencari korelasi antara harga mobil berdasarkan fitur-fitur dari dataset yang akan di gunakan
- Mencari solusi untuk mempermudah perusahaan dalam memprediksi harga mobil
- Mencari cara agar perusahaan dapat membaca visualisasi data dengan mudah

### Goals

- Analisa dataset untuk mencari korelasi harga mobil dengan fitur-fitur dataset
- Membuat sistem yang dapat memprediksi harga mobil dengan pendekatan predictive analtics dengan error sekecil mungkin
- Membuat visualisasi data yang mudah di baca oleh perusahaan

Solution statements :
  - Pada goal kedua, sistem prediksi akan dibangun menggunaan 3 model (Random Forest,KNN,dan Boosting)
  - Setiap model akan dilihat mana yang memiliki error terendah
  - Model yang memilki error terendah akan dilakukan tuning parameter untuk menemukan error terkeci dari model yang di pilih


## Data Understanding
Dataset yang akan digunakan adalah dataset sebuah mobil bekas yan terdiri dari 8 kolam dan 4341 data mobil bekas dari tahun 1992 - 2020. Dataset ini mencakup harga,tahun,dan fitur-fitur mobil. Data yang digunakan telah di analysis. Tidak terdapat missing value,duplicate dllnya pada data. Namun,terdapat anomali pada data km_driven yang menunjukan nilai terendah adalah 1. Nilai anomali ini di drop. Screenshoot data anomali tersebut dapat telihat di kriteria tambahan statistik variabel numerik data dibawah.
 Sumber dataset : [Vehicle dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

### Variabel-variabel pada Vehicle dataset adalah sebagai berikut:
- name : kolam yang berisi nama-nama mobil tertentu
- year : kolam yang berisi tahun ketika mobil tersebut dibeli
- selling_price : kolam yang berisi harga jual mobil tersebut
- km_driven : kolam yang berisi kilometer dari mobil tersebut ketika di jual bekas
- fuel : kolam yang berisi jenis bahan bakar yang digunakan mobil tersebut
- seller_type : kolam yang berisi penjual mobil tersebut ( dialer atau penjual individu )
- transmission : kolam yang berisi transmisi mobil antara manual atau automatic
- owner : informasi pemilik mobil yang menjual ( pemilik pertama atau kedua atau lebih )


### Data load

![Data Loading](https://github.com/elfander/dicoding_machine_learning/blob/main/data_loading.png?raw=true)
---
Data yang telah di load menampilkan 8 kolam yang berisi fitur-fitur pada sebuah mobil. Data juga menampilkan 4339 baris data. 

### Exploratory Data Analysis - Deskripsi Variabel

![Deskripsi Variabel](https://github.com/elfander/dicoding_machine_learning/blob/main/deskripsi_variabel.png?raw=true)
---
Pada analasisis deksripsi variabel, didapati bahwa terdapat 5 kolam dengan tipe data object. Dan tipe data numerik (int64) sebanyak 3 data. Data yang akan kita prediksi adalah selling_price dengan tipe data int64

### Statistik variabel data numerik

![Ddescribe mobil](https://github.com/elfander/dicoding_machine_learning/blob/main/mobil%20describe.png?raw=true)
---
Didapati informasi bahwa, Ternyata pada fitur km_driven terdapat anomali data. Karena jumlah min nya adalah 1. Mustahil mobil bekas baru berjalan 1 KM. Namun jangan langsung di hapus data tersebut. Dilihat dulu apakah data lain juga anomali.

### Pengecekan km_driven terendah pada data mobil lain

![KM_DRIVEN](https://github.com/elfander/dicoding_machine_learning/blob/main/km.png?raw=true)
---
Didapati informasi bahwa terdapat 4 mobil dengan KM terendah. Namun, 101 adalah KM terendah yang dimiliki oleh Test Drive Car. Sedangkan data anomali satu,dimiliki oleh seccond Owner. Maka data dengan KM_DRIVEN 1 terseut bisa kita drop atau hapus. 

### Menangani Outliers untuk melihat data terluar

Outliers adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ia adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. Outliers akan di deteksi menggunakan teknik visualisasi data (boxplot).

![Outliers km_driven](https://github.com/elfander/dicoding_machine_learning/blob/main/outliers%20km_driver.png?raw=true)
---
![outliers price](https://github.com/elfander/dicoding_machine_learning/blob/main/outliers%20selling%20price.png?raw=true)
---
![outliers year](https://github.com/elfander/dicoding_machine_learning/blob/main/outliers%20years.png?raw=true)
---
Pada data-data diatas, terdapat outliers di fitur numerik year,km_driven,dan price. Maka dari itu, Outliers tersebut di dalam notebook diatasi dengan menggunakan metode IQR.

### visualisasikan fitur paling berpengaruh terhadap price berdasarkan dataset (untuk mencapai problem statements ke 3)

![abg price fuel](https://github.com/elfander/dicoding_machine_learning/blob/main/price%20fuel.png?raw=true)
---
Pada diagram diatas dapat terlihat bahwa, fitur categorical fuel pada jenis Diesel, rata-rata harganya lebih tinggi dibandingkan yang lain.

![price owner](https://github.com/elfander/dicoding_machine_learning/blob/main/price%20owner.png?raw=true)
---
Pada diagram diatas dapat terlihat bahwa, fitur categorical owner dengan jenis test_drive, rata-rata harganya lebih tinggi dibandingkan yang lain. Hal itu masuk akal, karena test drive yang bukanlah mobil utama dalam pemakaian

![year price](https://github.com/elfander/dicoding_machine_learning/blob/main/price%20dan%20tahun.png?raw=true)
---
Pada diagram diatas dapat terlihat bahwa, fitur numerical tahun sangat mempengaruhi harga, rata-rata harganya lebih tinggi jika tahun pembelian mobil tersebut tinggi.

### korelasi antara fitur numerik

![korelasi](https://github.com/elfander/dicoding_machine_learning/blob/main/korelasi.png?raw=true)
---

![Correlation Matrix](https://github.com/elfander/dicoding_machine_learning/blob/main/correlation%20matrix.png?raw=true)
---

pada data-data diatas, kesimpulanya adalah, Semakin tinggi tahun, semakin tinggi juga penjualan. Korelasi km_driven dan harga tidak begitu mempengaruhi. Fuel pada mobil juga mempengaruhi rata-rata harga.
 

## Data Preparation
Data preparation yang dilkaukan adalah dengan melakukan encoding pada fitur categorical dengan tipe data object. 
lalu melakukan standarisasi nilai(year) dan membagi antara data test dan data train.

### Encoding fitur categorical

Tujuan dari encoding fitur categorical adalah untuk mengubah isi data yang berjenis categorical ke bentuk numerical. Hal itu karena algoritma machine learning tidak bisa melakukan kalkukasi terhadap fitur categorical. Fitur categorical yang di encoding antara lain :  ['name', 'fuel', 'seller_type', 'transmission', 'owner'].

![encoding](https://github.com/elfander/dicoding_machine_learning/blob/main/encoding.png?raw=true)
---
Pada data diatas,dapat terlihat bahwa penerapan encoding berhasil dilakukan. Setiap isi data dari fitur categorical sudah berubah menjadi numerik. 

### Pembagian data train dan test

Tujuan dari membagi data uji (test data) dan data latih (training data) adalah untuk mengevaluasi dan memastikan bahwa model yang dibangun dapat menggeneralisasi dengan baik pada data yang belum pernah dilihat sebelumnya. Pada analisis ini, pembagian data uji adalah porsi 10% untuk data uji dan 90% untuk data latih. Berikut total data setelah di bagi :
Total # of sample in whole dataset: 3857
Total # of sample in train dataset: 3471
Total # of sample in test dataset: 386
data train adalah semua data yang telah di encoding dan data numerik pada dataset keculi fitur price.
data test adalah semua data price. Data price akan di prediksi dan data price adalah target.

### standarisasi
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Standarisasi pada fitur numerik dilakukan dengan memanfaatkan teknik StandarScaler dari library scikit learn. Data yang akan di standarisasi adalah data train.
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Pada analisis ini, target standarisasi adalah fitur year. Karena fitur numerik lainya seperti km_driven sudah di drop karena tidak berpengaruh besar terhadap harga atau price. 

![standarisasi](https://github.com/elfander/dicoding_machine_learning/blob/main/standarisasi.png?raw=true)
---
dapat terlihat fitur year setelah dilakukan standarisasi, STD nya bernilai 1 dan mean bernilai 0.


## Modeling

Pada notebook, terdapat 3 model yang digunakan untuk prediksi. Tapi dari 3 model ini akan dilihat model mana yang memberikan error rate terendah dan model mana yang akan memberikan gap kecil antara data test dan data train. Model yang di training antara lain : random forrest, KNN, dan Boosting algorithm. Ketiga model ini memanfaatkan library scikit learn. Ketiga model ini akan di evaluasi menggunakan matrix MSE. Sehingga di dalam program ketika training, MSE dilibatkan.
Sebelum melakukan modeling, siapkan dataframe terlebih dahulu. Dataframe ini akan berisi hasil train dan hasil test dari ketiga model. Bentuk datafamre tersebut seperti ini :

![MSE](https://github.com/elfander/dicoding_machine_learning/blob/main/MSE.png?raw=true)
---
Gambar diatas menunjukan bahwa gap atau perbedaan jarak antara error dari data train dan test.

### Model KNN
  
parameter yang digunakan pada KNN adalah n_neighbors=6. Menunjukkan bahwa model akan mempertimbangkan 6 tetangga terdekat pada fitur selain price untuk memprediksi nilai price, model akan mempelajari polda dalam data-data ini. Data yang dimaksud adalah x_train sebagai fitur-fitur pada dataset dan y_train yang merupakan fitur price atau harga. Pada prediksi ini, model menggunakan k = 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik. 


**kekurangan KNN**

KNN mempunyai kekurangan terhadap data dengan fitur yang banyak. kekurangan ini disebut dengan Curse of Dimensionality.
Curse of Dimensionality dalam KNN adalah masalah di mana peningkatan jumlah dimensi menyebabkan jarak antar titik data menjadi serupa dan kurang informatif, yang membuat sulit untuk menentukan tetangga terdekat yang relevan, meningkatkan kompleksitas komputasi, menyebabkan data menjadi sparse, dan rentan terhadap overfitting, sehingga menurunkan efektivitas KNN dalam menangani data berdimensi tinggi.


**kelebihan KNN**

Seperti yang kita tau Algoritma KNN mudah diimplementasikan. Bahkan di para praktisi ML di komunitas dan forum mengatakan, untuk menerapkan KNN tidak memerlukan asumsi tentang distribusi data.




### Model Random forest
  
Korelasi atau pola pada data x_train sebagai fitur selain price akan digunakan pada paramter-parameter random forest. Begitu juga dengan fitur y_train sebagai fitur target atau fitur price. Parameter tersebut dibentuk dari library scikit-learn dengan nama RandomForestRegressor. Parameter yang digunakan dalam Random Forest ini antara lain :

- n_estimators=50: Jumlah pohon dalam algoritma Random Forrest. Dalam hal ini, model akan menggunakan 50 decission tree.
- max_depth=16: Kedalaman maksimum setiap tree. Ini berarti setiap tree bisa memiliki kedalaman hingga 16 lapisan, yang membantu dalam mencegah overfitting dengan membatasi kompleksitas tree.
- random_state=55: Parameter ini memastikan hasil yang dihasilkan konsisten antara run dengan memberikan seed yang tetap untuk generator angka acak.
- n_jobs=-1: Menentukan jumlah pekerjaan yang berjalan secara paralel. -1 berarti menggunakan semua prosesor yang tersedia untuk mempercepat proses.

**kekurangan random forest**

Seperti yang terlihat diatas, parameter yang digunakan dalam melakukan training untuk random forest lebih banyak dibandingkan 2 model lainya. Ya itu adalah kekuranganya. Jika kita fokus dan memilih model ini, model ini akan menguras waktu dalam proses tuning parameter. 

**kelebihan random forest**

Karena menggunakan banyak pohon keputusan, model ini dapat menangani data dengan baik dan cenderung lebih stabil. Jika tuning paramater berhasil, model ini akan dapat menangani noise dan data yang hilang dengan baik. Tapi dengan catatan, tuning parameter harus betul-betul baik.


### Model Boosting Algorithm
Pada model boosting algotithm, metode yang digunakan adalah adaptive boosting. Salah satu metode adaptive boosting yang terkenal adalah AdaBoost. Pada analissi ini, penereapan adabost dengan memanggil AdaBoostRegressor dari scikit learn. 


Cara kerjanya adalah semua kasus dalam data latih memiliki weight atau bobot yang sama. Pada setiap tahapan, model akan memeriksa apakah observasi yang dilakukan sudah benar? Bobot yang lebih tinggi kemudian diberikan pada model yang salah sehingga mereka akan dimasukkan ke dalam tahapan selanjutnya. Proses iteratif ini berlanjut sampai model mencapai akurasi yang diinginkan. Dalam hal ini, data latih adalah semua data latih tidak terkecual price(y_train). 

Berikut parameter yang digunakan pada model Boosting Algorithm : 
- learning_rate=0.05: Ini adalah faktor pengali yang digunakan untuk mengontrol kontribusi setiap estimator individu. Nilai yang lebih kecil memperlambat laju pembelajaran, yang bisa membantu dalam menghindari overfitting.
- random_state=55: Parameter ini memastikan hasil yang dihasilkan konsisten antara run dengan memberikan seed yang tetap untuk generator angka acak.

**kekurangan adaboost**

Umumnya model ini rentan terhadap overfitting. Karena cara kerjanya yang sangat agreif terhadap weight atau bobot data train. Serta proses train pada adaboost ini dikenal lebih lama dibandingkan 2 lainya. 


**kelebihan Adaboost**

Sebagai mahasiswa yang belum begitu berpengalaman dalam ML, Sejujurnya saya belum atau tidak mengetahui kelebihan adaboost. Namun jika kita mengacu ke tuning parameter, Adaboost cukup sederhana dalam melakukan tuning. Karena parameter utamanya hanyalah learning_rate dan random_state. 2 parameter ini umum di temukan pada algoritma lainya di ML. 


## Evaluation
metrik evaluasi yang digunakan adalah MSE atau Mean Squared Error. Hasil MSE dari 3 model dapat dilihat menggunakan diagram seperti dibawah : 
- Hasil diagram pertama sbelum tuning parameter KNN
---
 ![10](https://github.com/elfander/dicoding_machine_learning/blob/main/10.png?raw=true)
---
-  Hasil diagram pertama stelah tuning parameter KNN
---
![6](https://github.com/elfander/dicoding_machine_learning/blob/main/6.png?raw=true)
---
Dari 2 grafik diatas, dapat terlihat bahwa error terendah dengan gap terkecil adalah KNN. Dan KNN sebelum dilakukan tuning parameter, medapatkan error rate yang tinggi. Setelah dilakukan tuning parameter, mendapatkan error rate yang sedikit lebih rendah dari yang sebelumnya.

Untuk mengujinya, Berikut prediksi menggunakan 3 model :
---
![prediksi](https://github.com/elfander/dicoding_machine_learning/blob/main/prediksi.png?raw=true)
---
dari hasil prediksi pengujian, gap antara prediksi dan y_true yang paling rendah adalah KNN. Sehingga problem statement dengan membuat sistem prediksi dipilih menggunakan KNN.

**Goals & problem statement yang sudah tercapai**

- Sebelumnya, problem stantement pertama adalah mencari korelasi antara data yang mempengaruhi harga. . Goal dari statement pertama adalah analisa dataset untuk mencari korelasi. Korelasi telah ditemukan. Dari data numerik, Harga paling dipejgaruhi oleh tahun. Didalam data numerik, juga terdapat km_driven yang mempengaruhi. Namun ternyata ketika dilakukan uji dengan correlation matrik, km_driven tidak begitu mempengaruhi, sehingga fitur km_driven di drop sebelum dilakukan training.
  
- Pada problem statement kedua, tujuanya adalah untuk Mencari solusi agar mempermudah perusahaan dalam memprediksi harga mobil. Goal dari statement ini adalah membuat sistem prediksi dengan pendekatan predictive analytics. Sistem preiksi sudah dibangun dan tertera di notebook. Model yang berhasil atau mendekati error dan gap paling kecil adalah KNN. Dapat terlihat pada gambar diatas ketika pengujian, KNN adalah model yang paling mendekati y_true sebagai data uji.

- Pada problem statement ketiga, tujuanya adalah Mencari cara agar perusahaan dapat membaca visualisasi data dengan mudah. Dan goal yang di lakukan sudah mecapai hal itu. Berikut visualisasi data berdasarkan price yang akan ditujukan untuk perusahaan

  **rata-rata harga jual dibandingkan tahun**
![correlation matrix](https://github.com/elfander/dicoding_machine_learning/blob/main/price%20dan%20tahun.png?raw=true)
---
Pada gambar diatas perusahan akan membaca bahwa harga penjualan mobil sangat dipengaruhi oleh tahun. Semakin tinggi tahun pembelian mobil bekas tersebut, maka semakin tinggi harga jualnya

  **rata-rata harga jual dibandingkan owner sebelumnya**

![jual dan fuel](https://github.com/elfander/dicoding_machine_learning/blob/main/price%20owner.png?raw=true)
---
Pada gambar diatas, perusahan akan membaca bahwa mobil dengan owner sebelumnya adalah test_drive, akan membuat rata-rata harga tinggi. Namun, semakin tinggi angka owner seperti 3 dan 4, maka semakin membuat harga mobil bekas turun.

**Rubrik/Kriteria Tambahan (Opsional)**: 

![rumus mse](https://qph.cf2.quoracdn.net/main-qimg-7bc0215acc561adfe5bb8920751eea2f)
source image :https://qph.cf2.quoracdn.net/main-qimg-7bc0215acc561adfe5bb8920751eea2f

Penjelasan rumus :
- 𝑁: Jumlah total pengamatan atau data point dalam dataset. Ini adalah ukuran populasi data yang digunakan untuk menghitung MSE.
- yi: Nilai sebenarnya atau nilai aktual dari pengamatan ke-𝑖Ini adalah nilai yang diobservasi dari dataset.
- (yi−ŷi)^2 : Kesalahan (error) untuk pengamatan ke-𝑖Ini adalah selisih antara nilai aktual dan nilai prediksi. Kuadrat dari kesalahan tersebut, Mengkuadratkan kesalahan membuat semua nilai positif (karena kesalahan bisa positif atau negatif) dan memperbesar pengaruh kesalahan yang lebih besar.
- ∑N-1=1: Simbol ini menunjukkan bahwa kita menjumlahkan kesalahan kuadrat untuk semua pengamatan dari 𝑖=1 hingga 𝑖=𝑁.
- 1/N :  Membagi jumlah kesalahan kuadrat dengan jumlah pengamatan (N) untuk mendapatkan rata-rata kesalahan kuadrat. Inilah mengapa metrik ini disebut "Mean Squared Error."


**---Ini adalah bagian akhir laporan---**
Refrensi

[1] Belgiawan, P.F., Schmöcker, JD., Abou-Zeid, M. et al. Car ownership motivations among undergraduate students in China, Indonesia, Japan, Lebanon, Netherlands, Taiwan, and USA. Transportation 41, 1227–1244 (2014). https://doi.org/10.1007/s11116-014-9548-z

[2] Samruddhi, K., & Kumar, R. A. (2020). Used Car Price Prediction using K-Nearest Neighbor Based Model. International Journal of Innovative Research in Applied Sciences and Engineering, 4(2), 629–632. https://doi.org/10.29027/ijirase.v4.i2.2020.629-632
