# from crypt import methods
from turtle import left
from flask import Flask, render_template
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline
import numpy as np
import os
import array
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from flask import Flask, render_template, request, url_for, redirect, jsonify
from flask_mysqldb import MySQL
from sqlalchemy import null

app = Flask(__name__)

#######################################################
#loading excel file ke dalam 1 variabel 'df'
# df = pd.read_excel (r'data\default.xlsx', sheet_name='UKM Jasa')
df = pd.read_excel (r'data\default.xlsx')
df.index = np.arange(1, len(df) + 1)
# df.index += 1

df_cleaning = df
data = df_cleaning.iloc[:,[7,18,27,29,30,31,32,33,34,35,36,37]]
data_cluster = data

df_transformasi = df_cleaning
df_transformed = df_cleaning

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cleaning',methods=['GET', 'POST'])
def cleaning():
    global df
    global data
    global df_cleaning
    global data_cluster

    if request.method == 'POST':
        load = request.files['file']
        load.save(os.path.join('data','UKM Fashion & UKM Kerajinan - Data UKM Pendataan Verifikasi Periode II 2021-2022_Dinas PKU.xlsx'))
        
        # PREPROCESSING DATA 
        # CLEANING DATA
        df = pd.read_excel (r'data\UKM Fashion & UKM Kerajinan - Data UKM Pendataan Verifikasi Periode II 2021-2022_Dinas PKU.xlsx')
        df_cleaning = df
        df.index = np.arange(1, len(df) + 1)

        # df_cleaning = df_cleaning.dropna()
        # df_cleaning = df_cleaning.drop(labels = [1])
        
        # TAHAPAN SELEKSI DATA
        data_cluster = df_cleaning.iloc[:,[1,15,7,18,27,29,30,31,32,33,34,35,36,37]]
        data_cluster.columns = ['ref_oss','nama_usaha','pendidikan','tanggal_pendirian_usaha','kegiatan_usaha', 'tujuan_pemasaran', 'kepemilikan_tanah', 'sarana_media_elektronik', 'modal_bantuan_pemerintah', 'pinjaman', 'omset_pertahun', 'asuransi', 'tenaga_kerja_laki', 'tenaga_kerja_perempuan']
        # data_cluster.index += 1
        data_cluster.index = np.arange(1, len(data_cluster) + 1)

        data = df_cleaning.iloc[:,[7,18,27,29,30,31,32,33,34,35,36,37]]
        data.columns = ['pendidikan','tanggal_pendirian_usaha','kegiatan_usaha', 'tujuan_pemasaran', 'kepemilikan_tanah', 'sarana_media_elektronik', 'modal_bantuan_pemerintah', 'pinjaman', 'omset_pertahun', 'asuransi', 'tenaga_kerja_laki', 'tenaga_kerja_perempuan']
        print(data,df,df_cleaning,data_cluster)
    return render_template('cleaning.html',data_tabel=[data_cluster.to_html(classes="table table-bordered hover nowrap",table_id="cleannn")], raw = [df.to_html(classes="table table-bordered hover nowrap",table_id="clean")])

@app.route('/transformasi')
def transformasi():
    # DATA TRANSFORMASI
    global df_transformed
    global data

    df_transformasi = data
    ##Tranformasi kolom Pendidikan
    pendidikan_transformed = pd.get_dummies(data.pendidikan)
    


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
    df_transform = pd.concat([pendidikan_transformed,df_umur_usaha,kegiatan_usaha_transformed,tujuan_pemasaran_transformed, kepemilikan_tanah_transformed,sarana_media_elektronik_transfromed,modal_bantuan_pemerintah_transformed,pinjaman_transfromed,omset_pertahun_transformed,asuransi_transformed,df_tenagakerja_laki,df_tenagakerja_perempuan], axis='columns')
    df_transformed = df_transform
    print(df_transformed)
    return render_template('transformasi.html',data_transfomasi=[df_transformed.to_html(classes="table table-bordered hover nowrap",table_id="data")])

@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    var_cluster = request.form.get('titik_pusat')
    global df_transformed
    # global df_cleaning
    # global data
    global data_cluster

    ##dendogram
    plt.figure(figsize=(300, 50))
    plt.title("Dendograms")
    dend = shc.dendrogram(shc.linkage(df_transformed, method='average'))
    plt.savefig('static/images/dendogram.png', format='png', bbox_inches='tight')

    ##CLUSTERING
    clustering = AgglomerativeClustering(n_clusters= int(var_cluster), affinity='euclidean', linkage='average')
    cluster_result = clustering.fit_predict(df_transformed)

    ##PENGUJIAN SILHOUETTE
    silh_avg_score_ = silhouette_score(df_transformed, cluster_result)

    
    data_print_cluster = data_cluster
    
    ##PENAMBAHAN CLUSTER KE TABEL
    data_print_cluster['cluster'] = cluster_result

    new_X = data_print_cluster

    new_Y = pd.DataFrame(new_X)
    # clust0 = new_Y.apply(lambda x: True if x['cluster'] == 0 else False , axis=1)
    # clust1 = new_Y.apply(lambda x: True if x['cluster'] == 1 else False , axis=1)
    # clust2 = new_Y.apply(lambda x: True if x['cluster'] == 2 else False , axis=1)
    # clust3 = new_Y.apply(lambda x: True if x['cluster'] == 3 else False , axis=1)
    # clust4 = new_Y.apply(lambda x: True if x['cluster'] == 4 else False , axis=1)
    # clust5 = new_Y.apply(lambda x: True if x['cluster'] == 5 else False , axis=1)
    # clust6 = new_Y.apply(lambda x: True if x['cluster'] == 6 else False , axis=1)
    # clust7 = new_Y.apply(lambda x: True if x['cluster'] == 7 else False , axis=1)
    # clust8 = new_Y.apply(lambda x: True if x['cluster'] == 8 else False , axis=1)
    for clus in range(0, 9):
        globals()['clust%s' % clus] = new_Y.apply(lambda x: True if x['cluster'] == clus else False , axis=1)

    # jumlah0 = len(clust0[clust0 == True].index)
    # jumlah1 = len(clust1[clust1 == True].index)
    # jumlah2 = len(clust2[clust2 == True].index)
    # jumlah3 = len(clust3[clust3 == True].index)
    # jumlah4 = len(clust4[clust4 == True].index)
    # jumlah5 = len(clust5[clust5 == True].index)
    # jumlah6 = len(clust6[clust6 == True].index)
    # jumlah7 = len(clust7[clust7 == True].index)
    jumlah8 = len(clust8[clust8 == True].index)
    for clus in range(0, 9):
        globals()['jumlah%s' % clus] = len(globals()[f"clust{clus}"][globals()[f"clust{clus}"] == True].index)


    Data = {'Chart': [jumlah0,jumlah1,jumlah2,jumlah3,jumlah4,jumlah5,jumlah6,jumlah7,jumlah8]}
    
    diagram_pie = pd.DataFrame(Data,columns=['Chart'],index = ['Cluster 0','Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4','Cluster 5', 'Cluster 6', 'Cluster 7','Cluster 8'])

    diagram_pie.plot.pie(y='Chart',figsize=(8,8),autopct='%1.2f%%', startangle=70)
    plt.savefig('static/images/chart.png', format='png', bbox_inches='tight')

    return render_template('cluster.html',data_hasil=[data_print_cluster.to_html(classes="table table-bordered hover",table_id="data")],cluster_count=var_cluster,slh=silh_avg_score_)

if __name__ == '__main__':
    app.run(debug=True)