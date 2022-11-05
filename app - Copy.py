from turtle import left
from flask import Flask, render_template, request, url_for, redirect, jsonify
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline
import numpy as np
import array
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from flask_mysqldb import MySQL
from sqlalchemy import null

app = Flask(__name__)
#######################################################
#inisialisasi variabel penampung
df = null
data = null


# ##dendogram
# plt.title("Dendograms")
# dend = shc.dendrogram(shc.linkage(df_transformed, method='complete'))

# ##PENGUJIAN SILHOUETTE
# silh_avg_score_ = silhouette_score(df_transformed, cluster_result)
# #print(f'Silhouette Score : {silhouette_score(df_process_transformed, cluster_result)}')


#######################################################

@app.route('/')
def index():
    global df_open

    df_open = pd.read_excel (r'C:\Users\sefti\Documents\Skripsi\Data UKM Pendataan Verifikasi Periode II 2021-2022_Dinas PKU - tanpa NIK.xlsx', sheet_name='UKM Jasa')
    
    return render_template('index.html')

df = df_open

@app.route('/cleaning')
def cleaning():
    # PREPROCESSING DATA 
    # CLEANING DATA
    df_cleaning = df
    df_cleaning = df_cleaning.dropna()
    df_cleaning = df_cleaning.drop(labels = [1])
    # TAHAPAN SELEKSI DATA
    global data_clean
    data_clean = df_cleaning.iloc[:,[7,18,27,29,30,31,32,33,34,35,36,37]]
    data_clean.columns = ['pendidikan','tanggal_pendirian_usaha','kegiatan_usaha', 'tujuan_pemasaran', 'kepemilikan_tanah', 'sarana_media_elektronik', 'modal_bantuan_pemerintah', 'pinjaman', 'omset_pertahun', 'asuransi', 'tenaga_kerja_laki', 'tenaga_kerja_perempuan']

    return render_template('cleaning.html',data_tabel=[data_clean.to_html(classes="table table-bordered",table_id="cleannn")])

data = data_clean

@app.route('/transformasi')
def transformasi():

    # DATA TRANSFORMASI
    df_transformasi = data
    ##Tranformasi kolom Pendidikan
    pendidikan_transformed = pd.get_dummies(df_transformasi.pendidikan)

    ##Penghitungan Umur Usaha
    for index, row in df_transformasi.iterrows():
        df_transformasi.loc[index, 'umur_usaha'] = datetime.now().year - int ( row['tanggal_pendirian_usaha'][-4:])
    
    df_umur_usaha = df_transformasi['umur_usaha']

    ##Tranformasi kolom kegiatan usaha
    kegiatan_usaha_transformed = df_transformasi['kegiatan_usaha'].str.get_dummies(sep=', ')

    ##Tranformasi kolom tujuan pemasaran
    tujuan_pemasaran_transformed = df_transformasi['tujuan_pemasaran'].str.get_dummies(sep=', ')

    #transformasi kolom kepemilikan tanah
    kepemilikan_tanah_transformed = df_transformasi['kepemilikan_tanah'].str.get_dummies(sep=', ')

    #transformasi kolom sarana media elektronik
    sarana_media_elektronik_transfromed = df_transformasi['sarana_media_elektronik'].str.get_dummies(sep=', ')

    #transformasi kolom modal bantuan pemerintah
    modal_bantuan_pemerintah_transformed = pd.get_dummies(df_transformasi.modal_bantuan_pemerintah)

    #transformasi kolom pinjaman
    pinjaman_transfromed = df_transformasi['pinjaman'].str.get_dummies(sep=', ')

    #transformasi kolom omset pertahun
    omset_pertahun_transformed = pd.get_dummies(df_transformasi.omset_pertahun)

    #transformasi kolom asuransi
    asuransi_transformed = df_transformasi['asuransi'].str.get_dummies(sep=', ')

    #memasukkan kolom ke variabel untuk digabungkan
    df_tenagakerja_laki = df_transformasi['tenaga_kerja_laki']
    df_tenagakerja_perempuan = df_transformasi['tenaga_kerja_perempuan']

    #proses penyatuan hasil transformasi untuk di transformasi
    df_transformed = pd.concat([pendidikan_transformed,df_umur_usaha,kegiatan_usaha_transformed,tujuan_pemasaran_transformed, kepemilikan_tanah_transformed,sarana_media_elektronik_transfromed,modal_bantuan_pemerintah_transformed,pinjaman_transfromed,omset_pertahun_transformed,asuransi_transformed,df_tenagakerja_laki,df_tenagakerja_perempuan], axis='columns')


    return render_template('transformasi.html',data_transfomasi=[df_transformed.to_html(classes="table table-bordered",table_id="data")])

@app.route('/cluster')
def cluster():

    ##CLUSTERING
    clustering = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
    cluster_result = clustering.fit_predict(df_transformed)
    data_print_cluster = df_cleaning.iloc[:,[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]]
    ##PENAMBAHAN CLUSTER KE TABEL
    data_print_cluster['cluster'] = cluster_result
    return render_template('cluster.html',data_hasil=[data_print_cluster.to_html(classes="table table-bordered",table_id="data")])

if __name__ == '__main__':
    app.run(debug=True)