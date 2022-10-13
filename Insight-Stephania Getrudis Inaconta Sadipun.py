#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- Styling ---
from IPython.core.display import display, HTML, Javascript

color_map = ['#4d7c79', '#6fa3a7']

prompt = color_map[-1]
main_color = color_map[0]
strong_main_color = color_map[1]
custom_colors = [strong_main_color, main_color]

css_file = '''
div #notebook {
background-color: white;
line-height: 20px;
}

#notebook-container {
%s
margin-top: 2em;
padding-top: 2em;
border-top: 4px solid %s;
-webkit-box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5);
    box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5);
}

div .input {
margin-bottom: 1em;
}

.rendered_html h1, .rendered_html h2, .rendered_html h3, .rendered_html h4, .rendered_html h5, .rendered_html h6 {
color: %s;
font-weight: 600;
}

div.input_area {
border: none;
    background-color: %s;
    border-top: 2px solid %s;
}

div.input_prompt {
color: %s;
}

div.output_prompt {
color: %s; 
}

div.cell.selected:before, div.cell.selected.jupyter-soft-selected:before {
background: %s;
}

div.cell.selected, div.cell.selected.jupyter-soft-selected {
    border-color: %s;
}

.edit_mode div.cell.selected:before {
background: %s;
}

.edit_mode div.cell.selected {
border-color: %s;

}
'''

def to_rgb(h): 
    return tuple(int(h[i:i+2], 16) for i in [0, 2, 4])

main_color_rgba = 'rgba(%s, %s, %s, 0.1)' % (to_rgb(main_color[1:]))
open('notebook.css', 'w').write(css_file % ('width: 95%;', main_color, main_color, main_color_rgba, main_color,  main_color, prompt, main_color, main_color, main_color, main_color))

def nb(): 
    return HTML("<style>" + open("notebook.css", "r").read() + "</style>")
nb()


# # <h1 style="font-family: Trebuchet MS; padding: 12px; font-size: 48px; color: #4d7c79; text-align: center; line-height: 1.25;"><b>!!Perbandingan Akurasi Algoritma KNN dan Random Forest dalam memprediksi IPM!!<span style="color: #000000"></span></b><br><span style="color: #6fa3a7; font-size: 24px">Stephania Getrudis Inaconta Sadipun_Insight</span></h1>
# <hr>

# # 1. PENGENALAN

# # 1.1. Dataset Problems (Problem Scoping)

# - Dilansir dari Badan Pusat Statistik Indonesia, Indeks Pembangunan Manusia (IPM) Indonesia tahun 2021 mencapai 72,29, yang mana nilai ini mengalami peningkatan sebesar 0,35 poin (0,49%) dibandingkan capaian tahun sebelumnya (71,94). **Hal ini merupakan hal yang baik tetapi juga menjadi tantangan untuk kehidupan mendatang. Berdasarkan data dari UNDP (United Nations Development Program) yakni badan yang bergerak dengan tujuan untuk memberikan bantuan, terutama untuk meningkatkan pembangunan negara-negara berkembang, pada tahun 2022 terdapat 9 negara yang masuk pada ketagori dengan IPM rendah. Lalu bagaimana dengan Indonesia? apakah masih bisa mempertahankan dan meningkatkan IPM, ataukah justru menurun?**
# - **Untuk menjawab pertanyaan ini, perlu untuk melakukan tindak lanjut. Salah satu caranya adalah dengan melakukan prediksi IPM terlebih dahulu, sehingga hasil prediksi dapat digunakan menjadi acuan untuk ditindaklanjuti sesuai dengan apa yang dihasilkan pada prediksi tersebut. Dengan ini Indonesia bisa tetap mempertahankan IPM serta bisa meningkatkan IPM di Indonesia.**
# 
# - Adapun dataset yang digunakan berisi faktor-faktor yang mempengaruhi Indeks Pembangunan Manusia (IPM), dimana **Machine Learning akan memprediksi apakah seseorang tergolong dalam kategori IPM 'Low', 'Normal', 'High' atau 'Very High'**

# # 1.2. Model Machine Learning Yang Digunakan

# Ada 2 model/algoritma Machine Learning yang digunakan dalam perbandingan untuk mencari akurasi IPM terbaik, yaitu **K-Nearest Neighbors** dan **Random Forest**

# # 1.3. Deskripsi Dataset (Dataset Aqcuisison)

# <div style="line-height: 2; color: #000000; text-align: justify">
#     Terdapat <b>5 variabel</b> pada dataset ini yaitu :
#     <ul>
#         <li> <b>1 variabel kategori</b> dan</li>
#         <li> <b>4 variabel numerik</b></li>
#     </ul>
# </div>
# <div style="line-height: 2; color: #000000; text-align: justify">
#     <b>Struktur dari dataset :</b><br>
#     
# <table style="width: 100%">
# <thead>
# <tr>
# <th style="text-align: center; font-weight: bold; ">Nama Variabel</th>
# <th style="text-align: center; font-weight: bold; ">Deskripsi</th>
# <th style="text-align: center; font-weight: bold; ">Sample Data Variabel</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td><b>Harapan_lama_Sekolah (numerik)</b></td>
# <td>Seberapa lama seseorang ingin bersekolah <br> (dalam tahun)</td>
# <td>14.36; 13.90; ...</td>
# </tr>
# <tr>
# <td><b>Pengeluaran_Perkapita (numerik)</b></td>
# <td>Seberapa banyak pengeluaran seseorang tiap bulannya<br>(dalam rupiah/bulan)</td>
# <td>14922; 11059; ...</td>
# </tr>
# <tr>
# <td><b>Rerata_Lama_Sekolah (numerik)</b></td>
# <td>Rata-rata seseorang menempuh pendidikan <br> (dalam tahun)</td>
# <td>9.37; 11.30; ...</td>
# </tr>
# <tr>
# <td><b>Usia_Harapan_Hidup (numerik)</b></td>
# <td>Seberapa besar harapan hidup seseorang <br> (dalam tahun)</td>
# <td>64.83; 71.20; ...</td>
# </tr>
# <tr>
# <td><b>IPM (kategori)</b></td>
# <td>Kategori Indeks Pembangunan Manusia pada masyarakat Indonesia <br></td>
# <td>Low; Normal; High; Very-High</td>
# </tr>
# </tbody>
# </table>
#     

# # 2. MENGIMPORT LIBRARY

# In[12]:


# --- Importing Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import yellowbrick
import pickle

from matplotlib.collections import PathCollection
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.style import set_palette
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap


# Library yang digunakan adalah library untuk melakukan proses komputasi dataset (merupakan data terstruktur), melakukan proses visualisasi data dan menerapkan model algoritma K-Nearest Neighbors dan Random Forest

# # 3. MEMBACA DATASET

# In[13]:


# --- Mengimport Dataset ---
dataset = pd.read_csv("IPM.csv")

# --- Membaca Dataset---
dataset.head()


# Variabel IPM berisi data-data berbentuk string dengan tipe data object, sedangkan variabel lainnya berupa data numerik dengan tipe data float64 dan int64. Hal ini perlu diperhatikan, karena data pada variabel IPM tidak akan bisa dicari korelasinya dengan variabel lainnya (menggunakan scatterplot dan heatmap) jika masih berbentuk string dengan tipe data object. Oleh karena itu, kita perlu melakukan transformasi data untuk mengubah isi data IPM menjadi numerik dan mengubah tipe datanya menjadi float64 (Low : 0, Normal : 1, High : 2 dan Very-High : 3).
# 
# Penting untuk dicari nilai korelasi antar variabel sehingga bisa dilihat variabel mana yang sangat memberikan pengaruh pada IPM dan mana yang tidak terlalu memberikan pengaruh. Variabel yang tidak terlalu memberikan pengaruh bisa dihapus (dimensi reduction - feature selection). Feature selection ini dilakukan untuk menghindari overfitting.

# In[14]:


# --- Mengeluarkan Info Dataset ---
print('======= Info Dataset =======')
print('Total Rows    : ', dataset.shape[0])
print('Total Columns : ', dataset.shape[1])

# --- Mengeluarkan Info Detail Dataset ---
print('\n======= Detail Info Dataset =======')
dataset.info(memory_usage = False)


# Dapat dilihat bahwa dataset memiliki jumlah kolom sebanyak 2196 dan jumlah baris sebesar 5 yang mencakup 4 variabel x dan 1 (IPM) variabel y, beserta dengan tipe datanya. Tipe data pada variabel IPM adalah object. Hal ini masih berhubungan dengan perubahan string menjadi numerik yang dijelaskan sebelumnya, data pada variabel IPM masih tetap belum bisa dicari nilai korelasinya dengan variabel lain (menggunakan scatterplot dan heatmap) walau sudah berubah menjadi numerik, karena tipe datanya masih berupa object(string), maka perlu untuk melakukan transformasi data untuk mengubah tipe datanya menjadi float64 seperti 3 variabel lainnya.
# 
# Selain itu, isi data perlu diubah menjadi numerik dan juga tipe data pada variabel IPM perlu diubah menjadi float64 karena model akan lebih mudah mengenal dan belajar dengan data-data numerik untuk bisa mendapatkan hasil prediksi, dan juga jika tidak dilakukan kedua transformasi data ini, model beresiko tidak dapat belajar dengan baik dan tidak bisa mengeluarkan prediksi.

# # 4. DATA EXPLORATION

# # 4.1. Handling Missing Values (Preprocessing Data - Data Cleaning)

# In[15]:


#--- Menghapus missing values ---
dataset.isnull().sum()


# Terdapat missing values untuk semua variabel, jadi perlu untuk dihilangkan karena missing values memungkinkan terjadinya kesalahan dalam interpretasi hasil analisis.

# # 4.2. Melakukan Perubahan Pada Data (Preprocessing Data - Data Transformation)

# In[16]:


#--- Melakukan trasnfomasi data yaitu :
#--- 1. Mengganti isi dari variabel IPM yang awalnya berupa kata-kata menjadi numerik 
dataset['IPM'] = dataset['IPM'].replace({'Low':0,'Normal':1,'High':2,'Very-High':3})
dataset.head()


# In[17]:


#--- 2. Mengganti tipe data variabel IPM dari object menjadi float64
print('======= Info Dataset =======')
print('Total Rows    : ', dataset.shape[0])
print('Total Columns : ', dataset.shape[1])

print('\n======= Detail Info Dataset =======')
dataset['IPM'] = dataset['IPM'].astype('float64')
dataset.info(memory_usage = False)


# Setelah dilakukan transofrmasi data, dapat dilihat bahwa isi variabel IPM sudah berubah menjadi numerik dan tipe datanya juga sudah berubah menjadi float64.

# # 4.2. Visualisasi Variabel Kategori

# In[18]:


# --- Visualisasi variabel IPM ---
sns.countplot(dataset['IPM'], color='#447170')


# Dapat dilihat bahwa jumlah IPM 0 atau Low dan 3 atau Very-High sangat sedikit yaitu kurang dari 200 orang dibandingkan yang lainnya.

# # 4.3. Visualisasi Variabel Numeric

# In[19]:


# --- Visualisasi variabel 'Harapan_Lama_Sekolah'
f = plt.figure(figsize=(20,4))

f.add_subplot(1,2,1)
sns.distplot(dataset['Harapan_Lama_Sekolah'], color='#4d7c79')

f.add_subplot(1,2,2)
sns.boxplot(dataset['Harapan_Lama_Sekolah'], color='#4d7c79')

# --- Visualisasi variabel IPM berdasarkan 'Harapan_Lama_Sekolah'
f = plt.figure(figsize=(10,4))

f.add_subplot(1,2,1)
sns.swarmplot(x=dataset['IPM'], y=dataset['Harapan_Lama_Sekolah'], color='#4d7c79')

f.add_subplot(1,2,2)
sns.boxplot(x=dataset['IPM'], y=dataset['Harapan_Lama_Sekolah'], color='#4d7c79')


# - Dapat dilihat pada output diatas, kisaran angka harapan lama sekolah (dilihat pada histogram dan boxplot bagian atas) ada pada 2.5 sampai 17.5 tahun dengan banyak data oulier (data outlier tidak akan dihapus karena pada program ini, kita tidak akan mencari nilai mean, yang mana nilai mean mempengaruhi outlier, yang mana akan menampilkan bias pada hasil akhir. Outlier juga tidak akan dihapus karena program ini akan menerapkan model Random Forest yang tahan terhadap data outlier). 
# - Sedangkan pengaruhnya terhadap IPM menunjukan bahwa orang-orang dengan harapan sekolah (dilihat pada scatterplot dan boxplot dibagian bawah) dibawah 4 - dibawah 14 tahun memiliki IPM yang rendah (Low/0.0). Orang-orang dengan harapan lama sekolah 11 - 15 tahun memiliki IPM yang normal (1.0), orang-orang dengan harapan lama sekolah 11 - 17 tahun memiliki IPM yang tinggi (High/2.0) dan orang-orang dengan harapan lama sekolah 13 - 18 tahun memiliki IPM yang sangat tinggi (Very-High/3.0)

# In[20]:


# --- Visualisasi variabel 'Pengeluaran_Perkapita'
f = plt.figure(figsize=(20,4))

f.add_subplot(1,2,1)
sns.distplot(dataset['Pengeluaran_Perkapita'], color='#4d7c79')

f.add_subplot(1,2,2)
sns.boxplot(dataset['Pengeluaran_Perkapita'], color='#4d7c79')

# --- Visualisasi variabel IPM berdasarkan 'Pengeluaran_Perkapita'
f = plt.figure(figsize=(10,4))

f.add_subplot(1,2,1)
sns.swarmplot(x=dataset['IPM'], y=dataset['Pengeluaran_Perkapita'], color='#4d7c79')

f.add_subplot(1,2,2)
sns.boxplot(x=dataset['IPM'], y=dataset['Pengeluaran_Perkapita'], color='#4d7c79')


# - Dapat dilihat pada output diatas, kisaran angka pengeluaran perkapita (dilihat pada histogram dan boxplot bagian atas) ada pada angka dibawah 5000 sampai 25000 rupiah per bulan dengan banyak data oulier. 
# - Sedangkan pengaruhnya terhadap IPM menunjukan bahwa orang-orang dengan pengeluaran perkapita per bulan (dilihat pada scatterplot dan boxplot dibagian bawah) dibawah 5000 - diatas 7500 rupiah memiliki IPM yang rendah (Low/0.0). Orang-orang dengan pengeluaran perkapita diatas 5000 - 12500 rupiah memiliki IPM yang normal (1.0), orang-orang dengan pengeluaran perkapita diatas 7500 - diatas 17500 rupiah memiliki IPM yang tinggi (High/2.0) dan orang-orang dengan pengeluaran perkapita diatas 12500 - diatas 22500 rupiah memiliki IPM yang sangat tinggi (Very-High/3.0)

# In[21]:


# --- Visualisasi variabel 'Rerata_Lama_Sekolah'
f = plt.figure(figsize=(20,4))

f.add_subplot(1,2,1)
sns.distplot(dataset['Rerata_Lama_Sekolah'], color='#4d7c79')

f.add_subplot(1,2,2)
sns.boxplot(dataset['Rerata_Lama_Sekolah'], color='#4d7c79')

# --- Visualisasi variabel IPM berdasarkan 'Rerata_Lama_Sekolah'
f = plt.figure(figsize=(10,4))

f.add_subplot(1,2,1)
sns.swarmplot(x=dataset['IPM'], y=dataset['Rerata_Lama_Sekolah'], color='#4d7c79')

f.add_subplot(1,2,2)
sns.boxplot(x=dataset['IPM'], y=dataset['Rerata_Lama_Sekolah'], color='#4d7c79')


# - Dapat dilihat pada output diatas, kisaran angka rata-rata lama sekolah (dilihat pada histogram dan boxplot bagian atas) ada pada angka dibawah 0 - 14 tahun dengan banyak data oulier.
# - Sedangkan pengaruhnya terhadap IPM menunjukan bahwa orang-orang dengan rata-rata lama sekolah (dilihat pada scatterplot dan boxplot dibagian bawah) dibawah 2 - 7 tahun memiliki IPM yang rendah (Low/0.0). Orang-orang dengan rata-rata lama sekolah diatas 4 - dibawah 11 tahun memiliki IPM yang normal (1.0), orang-orang dengan rata-rata lama sekolah 7 - 11 tahun memiliki IPM yang tinggi (High/2.0) dan orang-orang dengan rata-rata lama sekolah 11 tahun - diatas 12 tahun memiliki IPM yang sangat tinggi (Very-High/3.0)

# In[22]:


# --- Visualisasi variabel 'Usia_Harapan_Hidup'
f = plt.figure(figsize=(20,4))

f.add_subplot(1,2,1)
sns.distplot(dataset['Usia_Harapan_Hidup'], color='#4d7c79')

f.add_subplot(1,2,2)
sns.boxplot(dataset['Usia_Harapan_Hidup'], color='#4d7c79')

# --- Visualisasi variabel IPM berdasarkan 'Usia_Harapan_Hidup'
f = plt.figure(figsize=(10,4))

f.add_subplot(1,2,1)
sns.swarmplot(x=dataset['IPM'], y=dataset['Usia_Harapan_Hidup'], color='#4d7c79')

f.add_subplot(1,2,2)
sns.boxplot(x=dataset['IPM'], y=dataset['Usia_Harapan_Hidup'], color='#4d7c79')


# - Dapat dilihat pada output diatas, kisaran angka usia harapan hidup (dilihat pada histogram dan boxplot bagian atas) ada pada angka dibawah 55 - 80 tahun dengan banyak data oulier.
# - Sedangkan pengaruhnya terhadap IPM menunjukan bahwa orang-orang dengan usia harapan hidup (dilihat pada scatterplot dan boxplot dibagian bawah) 55 - 72 tahun memiliki IPM yang rendah (Low/0.0). Orang-orang dengan usia harapan hidup 58 - diatas 75 tahun memiliki IPM yang normal (1.0), orang-orang  dengan usia harapan hidup sekolah 65 - diatas 75 tahun memiliki IPM yang tinggi (High/2.0) dan orang-orang  dengan usia harapan hidup 70 - diatas 75 tahun memiliki IPM yang sangat tinggi (Very-High/3.0)

# In[23]:


# --- Mencari nilai korelasi antar features(variabel) dengan heatmap ---
plt.figure(figsize=(15, 10))
sns.heatmap(dataset.corr(), annot=True, linewidths=0.1, cmap='gray')
plt.suptitle('Correlation Map of Numerical Variables')


# Pada heatmap diatas, dapat dilihat hubungan antar variabel 1 dengan lainnya, dimana semakin putih, hubungannya semakin kuat, dan warna putih menunjukan hubungan dengan dirinya sendiri. Dapat dilihat juga bahwa keempat variabel x memiliki korelitas yang cukup kuat terhadap IPM (mempengaruhi IPM). Harapan lama sekolah berkorelasi dengan IPM sebesar 64%, pengeluaran perkapita berkorelasi dengan IPM sebesar 82%, rata-rata lama sekolah berkorelasi dengan IPM sebesar 77% dan usia harapan hidup berkorelasi dengan IPM sebesar 64%, sehingga tidak ada variabel yang perlu dihapus karena semuanya memiliki korelasi yang kuat dengan IPM.

# # 5. MODELLING

# In[24]:


# --- Memisahkan variabel y ---
x = dataset.drop(['IPM'], axis=1)
y = dataset['IPM']


# Variabel y / dependent perlu dipisahkan sehingga model mengetahui yang mana x dan yang mana y.

# In[25]:


# --- Splitting Dataset ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=4)


# Dataset kemudian dibagi menjadi 2 bagian yaitu menjadi data training sebesar 85% dan data testing sebesar 15%.

# # 5.1. MENGIMPLEMENTASIKAN MODEL K-NEAREST NEIGHBORS

# In[26]:


# --- Mengimplementasikan model KNN ---
KNNClassifier = KNeighborsClassifier(n_neighbors=5)
KNNClassifier.fit(x_train, y_train)
y_pred_KNN = KNNClassifier.predict(x_test)


# Setelah data set dibagi, kemudian kita sudah bisa mengimplementasikan model KNN pada data training untuk melatih model. Parameter yang digunakan adalah n_neighbors yaitu jumlah tetangga terdekat sebanyak 5.

# # 5.2. MENGIMPLEMENTASIKAN MODEL RANDOM FOREST

# In[27]:


# --- Mengimplementasikan model Random Forest ---
RFclassifier = RandomForestClassifier(n_estimators=2000, random_state=1, max_leaf_nodes=40, min_samples_split=20)
RFclassifier.fit(x_train, y_train)
y_pred_RF = RFclassifier.predict(x_test)


# Dilanjutkan dengan mengimplementasikan model Random Forest pada data training untuk melatih model. Parameter yang digunakan adalah n_restimators yaitu jumlah decision tree sebanyak 2000, random_state sebanyak 1, max_leaf_nodes yaitu maksimal leaf node atau node paling akir dari decision tree sebanyak 40 dan min_sample_split atau minimal node yang diperlukan untuk melakukan split sebesar 20.

# # 6. EVALUATION

# Setelah mengimplementasikan model, maka selanjutnya akan dilakukan evaluasi terhadap kedua model.

# # 6.1. Menampilkan Akurasi dan Performa K-Nearest Neighbors

# In[28]:


# --- Menampilkan akurasi KNN ---
KNNACC = accuracy_score(y_pred_KNN, y_test)
print('Akurasi Model K-Nearest Neighbors : ')

# --- Menampilkan Classification Report KNN ---
print('Classification Report')
print(classification_report(y_test, y_pred_KNN))

# --- Menampilkan performa dari penggunaan KNN yaitu learning curve ---
print('Evaluasi Performa KNN')
fig, ((ax1)) = plt.subplots(1, figsize=(10, 5))

knnlc = LearningCurve(KNNClassifier, ax1, title='K-Nearest Neighbour Learning Curve')
knnlc.fit(x_train, y_train)
knnlc.finalize()

plt.tight_layout();


# Pada output diatas dapat dilihat, bahwa akurasi KNN dalam memprediksi IPM adalah sebesar 81%, dengan training score bermula dengan 86% dan menurun kemudain naik lagi menjadi 85%, sedangkan nilai cross validationnya bermula pada 79% kemudain turun dan naik kembali pada nilai 79%.

# In[29]:


# --- Menampilkan akurasi Random Forest ---
RFACC = accuracy_score(y_pred_RF, y_test)
print('Akurasi Model Random Forest Accuracy: ')

# --- Menampilkan Classification Report Random Forest ---
print('Classification Report')
print(classification_report(y_test, y_pred_RF))

# --- Menampilkan performa dari penggunaan Random Forest yaitu learning curve ---
print('Evaluasi Performa Random Forest')
fig, ((ax1)) = plt.subplots(1, figsize=(10, 5))

rcclc = LearningCurve(RFclassifier, ax1, title='Random Forest Learning Curve')
rcclc.fit(x_train, y_train)
rcclc.finalize()


# Pada output diatas dapat dilihat, bahwa akurasi Random Forest dalam memprediksi IPM adalah sebesar 96% yang mana nilai ini lebih tinggi dibandingkan dengan akurasi KNN, dengan training score bermula dengan 97% dan naik sedikit hingga mencapai diatas 98% lalu kemudian sedikit menurun secara perlahan tetapi masih tetap pada nilai diatas 98%, sedangkan nilai cross validationnya bermula pada 88% kemudain naik terus hingga mencapai 96%. Oleh karena itu, model yang akan digunakan untuk memprediksi IPM adalah model Random Forest.

# # 6.2. Menyimpan Model Terbaik Ke dalam Pickle File

# In[30]:


# --- Menyimpan model terbaik ---
file = open('IPM_Prediction_RF.pkl', 'wb')
pickle.dump(RFclassifier, file)


# Setelah selesai memilih model terbaik, selanjutnya model bisa disimpan ke dalam pickle file, sehingga bisa dipakai kapan saja.

# # 6.3. Uji Coba Model Dengan Akurasi Terbesar Dalam Memprediksi IPM

# In[32]:


# --- Menginput 4 variabel yang mempengaruhi variabel IPM ---
data = [[14.36, 9572, 9.37, 69.96]]           

# --- Melakukan prediksi IPM dengan model Random Forest ---
result = RFclassifier.predict(data)

# --- Mencetak hasil prediksi IPM dengan model Random Forest---
if result[0] == 0:
    print('IPM dinyatakan rendah')
elif result[0] == 1:
    print('IPM dinyatakan normal')
elif result[0] == 2:
    print('IPM dinyatakan tinggi')
elif result[0] == 3:
    print('IPM dinyatakan sangat tinggi')


# Setelah model disimpan dalam pickle file, model bisa diuji coba untuk memprediksikan IPM. Pada output diatas, terlihat prediksi dilakukan dengan menginputkan 4 variabel yang mempengaruhi IPM, kemdian hasil prediksi akan didapatkan setelah menerapkan model Random Forest.

# # 7. Kesimpulan

# - Berdasarkan training score dan cross validation pada kedua model, model Random Forest memiliki kurva yang cenderung naik, sehingga dikatakan performa model Ranfom Forest dalam memprediksi nilai IPM lebih baik dari pada model K-Nearest Neighbors. Hal ini juga menunjukan bahwa model Random Forest belajar/mempelajari  data dengan lebih baik.
# - Akurasi dengan model K-Nearest Neighbors adalah sebesar 81% dan akurasi dengan model Random Forest adalah sebesar 96%
# - Model Random Forest dikatakan sebagai model yang paling baik diantara kedua model ini dalam melakukan prediksi akurasi.
